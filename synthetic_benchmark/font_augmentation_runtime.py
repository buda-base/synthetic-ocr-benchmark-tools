#!/usr/bin/env python3
"""Generate, validate, and assign temporary font variants for page rendering."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from font_augmentation import AugmentationSpec, create_augmented_font, source_sha256
from font_augmentation_raster_qc import compare_rasters
from render_font_augmentation_audit import (
    coverage_font_row,
    load_probes,
    probe_stacks,
    render_sample,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROBES = SCRIPT_DIR / "data" / "font_augmentation" / "probes.txt"
HARD_SHAPING_REASONS = {
    "shape_error",
    "notdef",
    "dotted_circle",
    "empty_shape",
    "zero_ink",
    "missing_base_letter",
}


@dataclass(frozen=True)
class RuntimeVariant:
    variant_id: str
    path: Path
    specs: tuple[AugmentationSpec, ...]
    raster_qc: dict[str, object]
    provenance: tuple[dict[str, object], ...]

    @property
    def labels(self) -> str:
        return "|".join(spec.label for spec in self.specs)


def _sample_slant(rng: random.Random) -> float:
    draw = rng.random()
    if draw < 0.45:
        bounds = (0.0, -3.0)
    elif draw < 0.75:
        bounds = (-3.0, -7.0)
    elif draw < 0.95:
        bounds = (-7.0, -11.0)
    else:
        bounds = (-11.0, -14.0)
    return round(rng.uniform(*bounds), 1)


def _sample_stroke_value(rng: random.Random) -> float:
    if rng.random() < 0.5:
        return round(rng.uniform(-0.015, -0.005), 3)
    return round(rng.uniform(0.005, 0.030), 3)


def sample_augmentation_specs(rng: random.Random) -> tuple[AugmentationSpec, ...]:
    """Sample one reviewed combined policy, ordered safely for outline rewriting."""
    specs: list[AugmentationSpec] = []
    stroke_draw = rng.random()
    if stroke_draw < 0.20:
        specs.append(AugmentationSpec("vertical_stroke", _sample_stroke_value(rng)))
    elif stroke_draw < 0.40:
        specs.append(AugmentationSpec("horizontal_stroke", _sample_stroke_value(rng)))
    elif stroke_draw < 0.50:
        specs.extend(
            (
                AugmentationSpec("vertical_stroke", _sample_stroke_value(rng)),
                AugmentationSpec("horizontal_stroke", _sample_stroke_value(rng)),
            )
        )
    specs.extend(
        (
            AugmentationSpec("width", round(rng.uniform(0.80, 1.20), 3)),
            AugmentationSpec("slant", _sample_slant(rng)),
        )
    )
    return tuple(specs)


def _font_object(row: dict[str, object], source: Path, face_index: int) -> SimpleNamespace:
    return SimpleNamespace(
        basename=str(row["basename"]),
        font_abs_path=source,
        ttc_face_index=str(face_index) if face_index else "",
        ps_name=str(row.get("ps_name") or row.get("font_name") or row["basename"]),
        font_size_pt=float(row["font_size_pt"]),
        dpi=int(row.get("dpi") or 300),
    )


def _hard_shaping_regressions(source_row, variant_row, stacks: list[str]) -> list[dict[str, str]]:
    """Reject shaping failures while ignoring affine-sensitive placement heuristics."""
    from coverage_common import HarfbuzzShaper

    source_shaper = HarfbuzzShaper(source_row)
    variant_shaper = HarfbuzzShaper(variant_row)
    regressions: list[dict[str, str]] = []
    for stack in stacks:
        before = source_shaper.shape(stack)
        if not before["ok"]:
            continue
        after = variant_shaper.shape(stack)
        reasons = set(str(after["reason"]).split(";"))
        hard = sorted(
            reason
            for reason in reasons
            if reason in HARD_SHAPING_REASONS or reason.startswith("shape_error")
        )
        if hard:
            regressions.append({"stack": stack, "reason": ";".join(hard)})
    return regressions


def _apply_specs(
    source: Path,
    face_index: int,
    specs: tuple[AugmentationSpec, ...],
    *,
    cache_dir: Path,
    fontforge_bin: str,
) -> tuple[Path, tuple[dict[str, object], ...]]:
    current = source
    current_face = face_index
    provenance: list[dict[str, object]] = []
    for spec in specs:
        current, step = create_augmented_font(
            current,
            face_index=current_face,
            spec=spec,
            cache_dir=cache_dir,
            fontforge_bin=fontforge_bin,
        )
        current_face = 0
        provenance.append(step)
    return current, tuple(provenance)


def _combined_variant_id(
    source_hash: str,
    face_index: int,
    specs: tuple[AugmentationSpec, ...],
) -> str:
    payload = {
        "source_sha256": source_hash,
        "face_index": face_index,
        "specs": [{"operation": spec.operation, "value": spec.value} for spec in specs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _font_seed(seed: int, source_hash: str, face_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{source_hash}:{face_index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def generate_variants_for_font(
    row: dict[str, object],
    *,
    count: int,
    seed: int,
    cache_dir: Path,
    probes_path: Path = DEFAULT_PROBES,
    fontforge_bin: str = "fontforge",
    hb_view_bin: str = "hb-view",
) -> tuple[list[RuntimeVariant], list[dict[str, object]]]:
    """Generate exactly ``count`` accepted variants or raise after bounded retries."""
    source = Path(str(row["font_abs_path"])).resolve()
    face_index = int(str(row.get("ttc_face_index") or "0"))
    source_hash = source_sha256(source)
    rng = random.Random(_font_seed(seed, source_hash, face_index))
    probes = load_probes(probes_path)
    stacks = probe_stacks(probes)
    render_text = "\n".join(probes)
    font = _font_object(row, source, face_index)
    source_font_row = coverage_font_row(font)
    qc_dir = cache_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    baseline_png = qc_dir / f"{source_hash[:16]}_{face_index}_baseline.png"
    render_ok, render_error = render_sample(
        source_font_row,
        render_text,
        baseline_png,
        hb_view_bin=hb_view_bin,
        overwrite=False,
    )
    if not render_ok:
        raise RuntimeError(f"Could not render augmentation baseline for {source.name}: {render_error}")

    accepted: list[RuntimeVariant] = []
    rejected: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    max_attempts = max(count * 6, count + 5)
    for _attempt in range(max_attempts):
        if len(accepted) >= count:
            break
        specs = sample_augmentation_specs(rng)
        spec_key = tuple((spec.operation, spec.value) for spec in specs)
        if spec_key in seen:
            continue
        seen.add(spec_key)
        variant_id = _combined_variant_id(source_hash, face_index, specs)
        try:
            path, provenance = _apply_specs(
                source,
                face_index,
                specs,
                cache_dir=cache_dir / "fonts",
                fontforge_bin=fontforge_bin,
            )
            variant_font_row = coverage_font_row(font, path, 0)
            regressions = _hard_shaping_regressions(source_font_row, variant_font_row, stacks)
            if regressions:
                raise ValueError(f"hard shaping regressions: {regressions[:3]}")
            variant_png = qc_dir / f"{variant_id}.png"
            render_ok, render_error = render_sample(
                variant_font_row,
                render_text,
                variant_png,
                hb_view_bin=hb_view_bin,
                overwrite=False,
            )
            if not render_ok:
                raise ValueError(f"probe render failed: {render_error}")
            raster_qc = compare_rasters(baseline_png, variant_png)
            if not raster_qc["automatic_pass"]:
                raise ValueError(f"raster QC failed: {raster_qc['warnings']}")
            accepted.append(
                RuntimeVariant(
                    variant_id=variant_id,
                    path=path,
                    specs=specs,
                    raster_qc=raster_qc,
                    provenance=provenance,
                )
            )
        except Exception as exc:
            rejected.append(
                {
                    "variant_id": variant_id,
                    "specs": [spec.label for spec in specs],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if len(accepted) != count:
        raise RuntimeError(
            f"Generated only {len(accepted)}/{count} accepted variants for {source.name} "
            f"after {max_attempts} attempts"
        )
    return accepted, rejected


def assign_variants_to_rows(
    rows: list[dict[str, object]],
    variants: list[RuntimeVariant],
) -> list[dict[str, object]]:
    """Return rows assigned round-robin across a validated variant pool."""
    if not variants:
        raise ValueError("At least one runtime font variant is required")
    assigned: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: int(item["image_id"])):
        variant = variants[(int(row["image_id"]) - 1) % len(variants)]
        item = dict(row)
        item["render_font_abs_path"] = str(variant.path)
        item["render_ttc_face_index"] = 0
        item["font_augmentation_id"] = variant.variant_id
        item["font_augmentation_specs"] = variant.labels
        item["font_augmentation_source_file"] = str(row["font_abs_path"])
        item["font_augmentation_source_face_index"] = str(row.get("ttc_face_index") or "")
        assigned.append(item)
    return assigned


def prepare_font_variants(
    rows: list[dict[str, object]],
    *,
    variants_per_font: int,
    seed: int,
    cache_dir: Path,
    probes_path: Path = DEFAULT_PROBES,
    fontforge_bin: str = "fontforge",
    hb_view_bin: str = "hb-view",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate per-font pools and attach one temporary variant to every plan row."""
    if variants_per_font <= 0:
        raise ValueError("variants_per_font must be positive")
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["font_abs_path"]),
            str(row.get("ttc_face_index") or ""),
            str(row["basename"]),
        )
        groups.setdefault(key, []).append(row)

    prepared: list[dict[str, object]] = []
    manifest_fonts: list[dict[str, object]] = []
    for (_path, _face, basename), font_rows in sorted(groups.items()):
        variants, rejected = generate_variants_for_font(
            font_rows[0],
            count=variants_per_font,
            seed=seed,
            cache_dir=cache_dir / basename,
            probes_path=probes_path,
            fontforge_bin=fontforge_bin,
            hb_view_bin=hb_view_bin,
        )
        prepared.extend(assign_variants_to_rows(font_rows, variants))
        manifest_fonts.append(
            {
                "basename": basename,
                "source_font": str(font_rows[0]["font_abs_path"]),
                "source_face_index": str(font_rows[0].get("ttc_face_index") or ""),
                "requested_variants": variants_per_font,
                "accepted": [
                    {
                        "variant_id": variant.variant_id,
                        "specs": [spec.label for spec in variant.specs],
                        "raster_qc": variant.raster_qc,
                        "provenance": list(variant.provenance),
                    }
                    for variant in variants
                ],
                "rejected_attempts": rejected,
            }
        )
    prepared.sort(key=lambda row: int(row["image_id"]))
    return prepared, {
        "seed": seed,
        "variants_per_font": variants_per_font,
        "font_count": len(manifest_fonts),
        "fonts": manifest_fonts,
    }
