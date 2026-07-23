#!/usr/bin/env python3
"""Generate, validate, and assign temporary font variants for page rendering."""

from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from font_augmentation import AugmentationSpec, create_augmented_font, source_sha256
from font_augmentation_raster_qc import RASTER_QC_VERSION, compare_rasters
from font_variant_cache import DEFAULT_FONT_CACHE_URI, S3FontVariantCache
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
MAX_FONT_AUGMENTATION_VARIANTS = 12
FONT_AUGMENTATION_POLICY_VERSION = 2


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


def font_augmentation_policy(
    *,
    seed: int,
    count: int,
    probes_path: Path,
    max_attempts: int,
) -> tuple[str, dict[str, object]]:
    policy = {
        "version": FONT_AUGMENTATION_POLICY_VERSION,
        "raster_qc_version": RASTER_QC_VERSION,
        "seed": seed,
        "requested_variants": count,
        "max_attempts": max_attempts,
        "probes_sha256": hashlib.sha256(probes_path.read_bytes()).hexdigest(),
    }
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20], policy


def original_font_fallback(
    source: Path,
    source_hash: str,
    face_index: int,
    *,
    reason: str,
    attempted_variants: int,
    decision_uri: str = "",
) -> RuntimeVariant:
    provenance: dict[str, object] = {
        "operation": "original_font_fallback",
        "source": str(source),
        "reason": reason,
    }
    if decision_uri:
        provenance["decision_uri"] = decision_uri
    return RuntimeVariant(
        variant_id=f"original-{source_hash[:20]}-f{face_index:03d}",
        path=source,
        specs=(),
        raster_qc={
            "status": "original_font_fallback",
            "warnings": [],
            "attempted_variants": attempted_variants,
        },
        provenance=(provenance,),
    )


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
    remote_cache: S3FontVariantCache | None = None,
) -> tuple[list[RuntimeVariant], list[dict[str, object]]]:
    """Generate up to ``count`` safe variants, falling back to the source font."""
    if count > MAX_FONT_AUGMENTATION_VARIANTS:
        raise ValueError(
            f"At most {MAX_FONT_AUGMENTATION_VARIANTS} variants may be generated per font"
        )
    source = Path(str(row["font_abs_path"])).resolve()
    face_index = int(str(row.get("ttc_face_index") or "0"))
    source_hash = source_sha256(source)
    max_attempts = max(count * 6, count + 5)
    policy_key, policy = font_augmentation_policy(
        seed=seed,
        count=count,
        probes_path=probes_path,
        max_attempts=max_attempts,
    )
    if remote_cache is not None:
        decision = remote_cache.fetch_unaugmentable(
            source_hash=source_hash,
            face_index=face_index,
            policy_key=policy_key,
        )
        if decision is not None:
            rejected_attempts = list(decision.get("rejected_attempts") or [])
            return [
                original_font_fallback(
                    source,
                    source_hash,
                    face_index,
                    reason=str(decision["reason"]),
                    attempted_variants=int(
                        decision.get("policy", {}).get(
                            "max_attempts",
                            len(rejected_attempts),
                        )
                    ),
                    decision_uri=str(decision["s3_uri"]),
                )
            ], rejected_attempts
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
    for _attempt in range(max_attempts):
        if len(accepted) >= count:
            break
        specs = sample_augmentation_specs(rng)
        spec_key = tuple((spec.operation, spec.value) for spec in specs)
        if spec_key in seen:
            continue
        seen.add(spec_key)
        variant_id = _combined_variant_id(source_hash, face_index, specs)
        spec_payload = [
            {"operation": spec.operation, "value": spec.value}
            for spec in specs
        ]
        if remote_cache is not None:
            cached = remote_cache.fetch(
                basename=str(row["basename"]),
                source_hash=source_hash,
                face_index=face_index,
                variant_id=variant_id,
                specs=spec_payload,
                raster_qc_version=RASTER_QC_VERSION,
            )
            if cached is not None:
                accepted.append(
                    RuntimeVariant(
                        variant_id=variant_id,
                        path=cached.path,
                        specs=specs,
                        raster_qc=cached.raster_qc,
                        provenance=cached.provenance,
                    )
                )
                continue
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
            runtime_variant = RuntimeVariant(
                variant_id=variant_id,
                path=path,
                specs=specs,
                raster_qc=raster_qc,
                provenance=provenance,
            )
            if remote_cache is not None:
                remote_cache.store(
                    basename=str(row["basename"]),
                    source_hash=source_hash,
                    face_index=face_index,
                    variant_id=variant_id,
                    specs=spec_payload,
                    font_path=path,
                    raster_qc=raster_qc,
                    provenance=provenance,
                )
            accepted.append(runtime_variant)
        except Exception as exc:
            rejected.append(
                {
                    "variant_id": variant_id,
                    "specs": [spec.label for spec in specs],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not accepted:
        reason = (
            f"No safe augmented variant was accepted after {max_attempts} attempts"
        )
        decision_uri = ""
        if remote_cache is not None:
            decision_uri = remote_cache.store_unaugmentable(
                source_hash=source_hash,
                face_index=face_index,
                policy_key=policy_key,
                policy=policy,
                reason=reason,
                rejected_attempts=rejected,
            )
        accepted.append(
            original_font_fallback(
                source,
                source_hash,
                face_index,
                reason=reason,
                attempted_variants=max_attempts,
                decision_uri=decision_uri,
            )
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
    workers: int = 1,
    s3_cache_uri: str = DEFAULT_FONT_CACHE_URI,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate per-font pools and attach one temporary variant to every plan row."""
    if variants_per_font <= 0:
        raise ValueError("variants_per_font must be positive")
    if variants_per_font > MAX_FONT_AUGMENTATION_VARIANTS:
        raise ValueError(
            f"variants_per_font cannot exceed {MAX_FONT_AUGMENTATION_VARIANTS}"
        )
    if workers <= 0:
        raise ValueError("workers must be positive")
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
    remote_cache = (
        S3FontVariantCache(s3_cache_uri, cache_dir / "s3")
        if s3_cache_uri
        else None
    )

    def generate_group(group):
        (_path, face, basename), font_rows = group
        variants, rejected = generate_variants_for_font(
            font_rows[0],
            count=variants_per_font,
            seed=seed,
            cache_dir=cache_dir / f"{basename}__face_{face or '0'}",
            probes_path=probes_path,
            fontforge_bin=fontforge_bin,
            hb_view_bin=hb_view_bin,
            remote_cache=remote_cache,
        )
        return basename, font_rows, variants, rejected

    group_items = sorted(groups.items())
    executor = ThreadPoolExecutor(max_workers=min(workers, len(group_items)))
    futures = {executor.submit(generate_group, group): group for group in group_items}
    font_progress = tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Generate and validate font variants",
        unit="font",
    )
    for future in font_progress:
        basename, font_rows, variants, rejected = future.result()
        cache_hits = sum(
            1
            for variant in variants
            if any(step.get("operation") == "s3_cache" for step in variant.provenance)
        )
        font_progress.set_postfix_str(
            f"{basename} ({len(variants)}/{variants_per_font}, "
            f"{cache_hits} cached, {len(rejected)} rejected)"
        )
        prepared.extend(assign_variants_to_rows(font_rows, variants))
        manifest_fonts.append(
            {
                "basename": basename,
                "source_font": str(font_rows[0]["font_abs_path"]),
                "source_face_index": str(font_rows[0].get("ttc_face_index") or ""),
                "requested_variants": variants_per_font,
                "s3_cache_hits": cache_hits,
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
    executor.shutdown(wait=True)
    prepared.sort(key=lambda row: int(row["image_id"]))
    manifest_fonts.sort(
        key=lambda item: (str(item["basename"]), str(item["source_face_index"]))
    )
    return prepared, {
        "seed": seed,
        "variants_per_font": variants_per_font,
        "workers": workers,
        "s3_cache_uri": s3_cache_uri,
        "font_count": len(manifest_fonts),
        "fonts": manifest_fonts,
    }
