#!/usr/bin/env python3
"""Select and download reviewable blank pecha pages from BDRC pipeline results."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import boto3
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROTATED_EXIF_ORIENTATIONS = {5, 6, 7, 8}
THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Volume:
    w_id: str
    i_id: str
    version: str
    median_wh_ratio: float


@dataclass(frozen=True)
class Candidate:
    w_id: str
    i_id: str
    version: str
    filename: str
    source_key: str
    width: int
    height: int
    display_width: int
    display_height: int
    aspect_ratio: float
    exif_orientation: int | None
    v2_label: str
    v2_prob: float
    ldv1_nb_contours: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "blank_paper_review",
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--ratio-min", type=float, default=3.5)
    parser.add_argument("--ratio-max", type=float, default=4.5)
    parser.add_argument("--blank-prob-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--read-workers", type=int, default=32)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--dest-bucket", default=os.getenv("BEC_DEST_S3_BUCKET", "bec.bdrc.io"))
    parser.add_argument(
        "--source-bucket",
        default=os.getenv("BEC_SOURCE_S3_BUCKET", "archive.tbrc.org"),
    )
    parser.add_argument("--contact-sheet-size", type=int, default=100)
    parser.add_argument(
        "--volumes-parquet",
        type=Path,
        default=Path(
            "/home/eroux/BUDA/softs/buda-scripts/analysis/bec/"
            "db_output/final/volumes/volumes.parquet"
        ),
    )
    return parser.parse_args()


def s3_client() -> Any:
    client = getattr(THREAD_LOCAL, "s3_client", None)
    if client is None:
        client = boto3.client("s3", region_name=os.getenv("BEC_REGION"))
        THREAD_LOCAL.s3_client = client
    return client


def source_prefix(w_id: str, i_id: str) -> str:
    md5_prefix = hashlib.md5(w_id.encode("utf-8")).hexdigest()[:2]
    if i_id.startswith("I") and i_id[1:].isdigit() and len(i_id) == 5:
        suffix = i_id[1:]
    else:
        suffix = i_id
    return f"Works/{md5_prefix}/{w_id}/images/{w_id}-{suffix}/"


def artifact_key(job: str, volume: Volume) -> str:
    return (
        f"{job}/{volume.w_id}/{volume.i_id}/{volume.version}/"
        f"{volume.w_id}-{volume.i_id}-{volume.version}.parquet"
    )


def is_first_two_pages(filename: str) -> bool:
    stem = filename.rsplit(".", 1)[0]
    return stem.endswith("0001") or stem.endswith("0002")


def eligible_volumes(
    path: Path,
    *,
    ratio_min: float,
    ratio_max: float,
) -> tuple[list[Volume], int]:
    table = pq.read_table(
        path,
        columns=["w_id", "i_id", "version_name", "median_wh_ratio", "nb_imgs"],
    ).to_pydict()
    volumes: list[Volume] = []
    for w_id, i_id, version, ratio, image_count in zip(
        table["w_id"],
        table["i_id"],
        table["version_name"],
        table["median_wh_ratio"],
        table["nb_imgs"],
    ):
        if version is None or ratio is None or int(image_count or 0) <= 0:
            continue
        ratio_value = float(ratio)
        if ratio_min <= ratio_value <= ratio_max:
            volumes.append(Volume(str(w_id), str(i_id), str(version), ratio_value))
    return volumes, len(table["w_id"])


def read_parquet(bucket: str, key: str, columns: list[str]) -> tuple[dict[str, Any], Any]:
    payload = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    table = pq.read_table(io.BytesIO(payload), columns=columns)
    return table.to_pydict(), table.schema.metadata or {}


def read_dimensions(bucket: str, volume: Volume) -> dict[str, tuple[int, int]]:
    key = source_prefix(volume.w_id, volume.i_id) + "dimensions.json"
    payload = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    records = json.loads(payload)
    return {
        str(record["filename"]): (int(record["width"]), int(record["height"]))
        for record in records
    }


def stable_page_order(seed: int, volume: Volume, filename: str) -> str:
    payload = f"{seed}:{volume.w_id}:{volume.i_id}:{filename}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidates_for_volume(
    volume: Volume,
    *,
    dest_bucket: str,
    source_bucket: str,
    ratio_min: float,
    ratio_max: float,
    blank_prob_threshold: float,
    seed: int,
) -> list[Candidate]:
    try:
        v2, _metadata = read_parquet(
            dest_bucket,
            artifact_key("script_classification_v2", volume),
            ["img_file_name", "status", "label", "prob", "exif_orientation_tag"],
        )
        ldv1, _metadata = read_parquet(
            dest_bucket,
            artifact_key("ldv1", volume),
            ["img_file_name", "nb_contours", "ok"],
        )
        dimensions = read_dimensions(source_bucket, volume)
    except Exception:
        return []

    ldv1_rows = {
        str(filename): (bool(ok), int(nb_contours))
        for filename, ok, nb_contours in zip(
            ldv1["img_file_name"],
            ldv1["ok"],
            ldv1["nb_contours"],
        )
    }
    results: list[Candidate] = []
    for filename, status, label, prob, orientation in zip(
        v2["img_file_name"],
        v2["status"],
        v2["label"],
        v2["prob"],
        v2["exif_orientation_tag"],
    ):
        filename = str(filename)
        probability = float(prob or 0.0)
        if status != "ok" or is_first_two_pages(filename):
            continue
        if str(label) != "blank" and probability >= blank_prob_threshold:
            continue
        ldv1_row = ldv1_rows.get(filename)
        if not ldv1_row or not ldv1_row[0] or ldv1_row[1] != 0:
            continue
        if filename not in dimensions:
            continue
        width, height = dimensions[filename]
        orientation_value = int(orientation) if orientation is not None else None
        if orientation_value in ROTATED_EXIF_ORIENTATIONS:
            display_width, display_height = height, width
        else:
            display_width, display_height = width, height
        if display_height <= 0:
            continue
        ratio = display_width / display_height
        if not ratio_min <= ratio <= ratio_max:
            continue
        results.append(
            Candidate(
                w_id=volume.w_id,
                i_id=volume.i_id,
                version=volume.version,
                filename=filename,
                source_key=source_prefix(volume.w_id, volume.i_id) + filename,
                width=width,
                height=height,
                display_width=display_width,
                display_height=display_height,
                aspect_ratio=ratio,
                exif_orientation=orientation_value,
                v2_label=str(label),
                v2_prob=probability,
                ldv1_nb_contours=ldv1_row[1],
            )
        )
    results.sort(key=lambda page: stable_page_order(seed, volume, page.filename))
    return results


def select_candidates(
    volumes: list[Volume],
    *,
    args: argparse.Namespace,
    target_count: int,
) -> tuple[list[Candidate], dict[str, int]]:
    random.Random(args.seed).shuffle(volumes)
    primary: list[Candidate] = []
    extras: list[Candidate] = []
    scanned = 0
    batch_size = max(128, args.read_workers * 8)
    for start in range(0, len(volumes), batch_size):
        batch = volumes[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=args.read_workers) as executor:
            results = executor.map(
                lambda volume: candidates_for_volume(
                    volume,
                    dest_bucket=args.dest_bucket,
                    source_bucket=args.source_bucket,
                    ratio_min=args.ratio_min,
                    ratio_max=args.ratio_max,
                    blank_prob_threshold=args.blank_prob_threshold,
                    seed=args.seed,
                ),
                batch,
            )
            for pages in results:
                scanned += 1
                if pages:
                    primary.append(pages[0])
                    extras.extend(pages[1:])
        print(
            f"Scanned {scanned}/{len(volumes)} volumes; "
            f"{len(primary)} volume-spread candidates",
            flush=True,
        )
        if len(primary) >= target_count:
            break
    selected = primary[:target_count]
    if len(selected) < target_count:
        extras.sort(
            key=lambda page: hashlib.sha256(
                f"{args.seed}:extra:{page.w_id}:{page.i_id}:{page.filename}".encode("utf-8")
            ).hexdigest()
        )
        selected.extend(extras[: target_count - len(selected)])
    return selected, {
        "eligible_ratio_volumes": len(volumes),
        "scanned_volumes": scanned,
        "distinct_selected_volumes": len({(page.w_id, page.i_id) for page in selected}),
    }


def local_filename(candidate: Candidate) -> str:
    return f"{candidate.w_id}__{candidate.i_id}__{candidate.filename}"


def download_candidate(
    candidate: Candidate,
    *,
    source_bucket: str,
    originals_dir: Path,
) -> tuple[Candidate, Path | None, str | None]:
    destination = originals_dir / local_filename(candidate)
    if destination.exists() and destination.stat().st_size > 0:
        return candidate, destination, None
    temporary = destination.with_name(destination.name + ".part")
    try:
        payload = s3_client().get_object(
            Bucket=source_bucket,
            Key=candidate.source_key,
        )["Body"].read()
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return candidate, destination, None
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return candidate, None, repr(exc)


def download_selected(
    candidates: list[Candidate],
    *,
    args: argparse.Namespace,
) -> tuple[list[tuple[Candidate, Path]], list[str]]:
    originals_dir = args.out_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[tuple[Candidate, Path]] = []
    errors: list[str] = []
    attempted = 0
    position = 0
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        while len(downloaded) < args.count and position < len(candidates):
            needed = args.count - len(downloaded)
            batch = candidates[position : position + needed]
            position += len(batch)
            results = executor.map(
                lambda candidate: download_candidate(
                    candidate,
                    source_bucket=args.source_bucket,
                    originals_dir=originals_dir,
                ),
                batch,
            )
            for candidate, path, error in results:
                attempted += 1
                if path is not None:
                    downloaded.append((candidate, path))
                else:
                    errors.append(f"{candidate.source_key}: {error}")
                if attempted % 100 == 0 or len(downloaded) == args.count:
                    print(
                        f"Downloaded {len(downloaded)}/{attempted} attempted source pages",
                        flush=True,
                    )
    return downloaded, errors


def label_font(size: int = 14) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_contact_sheets(
    downloaded: list[tuple[Candidate, Path]],
    *,
    destination: Path,
    sheet_size: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    columns = 5
    rows = max(1, (sheet_size + columns - 1) // columns)
    cell_width = 400
    image_height = 100
    label_height = 34
    font = label_font()
    for sheet_index, start in enumerate(range(0, len(downloaded), sheet_size), 1):
        subset = downloaded[start : start + sheet_size]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows * (image_height + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for cell_index, (candidate, path) in enumerate(subset):
            x = (cell_index % columns) * cell_width
            y = (cell_index // columns) * (image_height + label_height)
            try:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((cell_width - 12, image_height - 8), Image.Resampling.LANCZOS)
                    paste_x = x + (cell_width - image.width) // 2
                    paste_y = y + (image_height - image.height) // 2
                    sheet.paste(image, (paste_x, paste_y))
            except Exception:
                draw.rectangle((x + 4, y + 4, x + cell_width - 4, y + image_height - 4))
                draw.text((x + 12, y + 40), "preview error", fill="black", font=font)
            sequence = start + cell_index + 1
            label = (
                f"{sequence:04d} {candidate.w_id}/{candidate.i_id} "
                f"r={candidate.aspect_ratio:.2f} p={candidate.v2_prob:.3f}"
            )
            draw.text((x + 8, y + image_height + 7), label, fill="black", font=font)
        sheet.save(destination / f"contact_sheet_{sheet_index:02d}.jpg", quality=90)


def write_outputs(
    downloaded: list[tuple[Candidate, Path]],
    errors: list[str],
    *,
    args: argparse.Namespace,
    selection_stats: dict[str, int],
) -> None:
    csv_path = args.out_dir / "blank_paper_samples.csv"
    fields = [
        "sequence",
        *Candidate.__dataclass_fields__.keys(),
        "source_s3_uri",
        "local_path",
        "size_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sequence, (candidate, path) in enumerate(downloaded, 1):
            row = asdict(candidate)
            row.update(
                {
                    "sequence": sequence,
                    "source_s3_uri": f"s3://{args.source_bucket}/{candidate.source_key}",
                    "local_path": str(path.relative_to(args.out_dir)),
                    "size_bytes": path.stat().st_size,
                }
            )
            writer.writerow(row)
    manifest = {
        "selection": {
            "requested_count": args.count,
            "downloaded_count": len(downloaded),
            "distinct_downloaded_volumes": len(
                {(candidate.w_id, candidate.i_id) for candidate, _path in downloaded}
            ),
            "seed": args.seed,
            "aspect_ratio_width_over_height": [args.ratio_min, args.ratio_max],
            "v2_blank_rule": (
                f"status == ok and (label == blank or prob < {args.blank_prob_threshold})"
            ),
            "ldv1_confirmation": "ok == true and nb_contours == 0",
            "excluded_pages": "filenames ending 0001 or 0002",
            "sampling": "one page per shuffled volume before any second page",
            "volume_prefilter": (
                f"median_wh_ratio between {args.ratio_min} and {args.ratio_max} "
                f"from {args.volumes_parquet}"
            ),
            **selection_stats,
        },
        "storage": {
            "script_classification_bucket": args.dest_bucket,
            "source_bucket": args.source_bucket,
            "output_directory": str(args.out_dir),
            "total_downloaded_bytes": sum(path.stat().st_size for _candidate, path in downloaded),
        },
        "download_errors": errors,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not 0 < args.ratio_min <= args.ratio_max:
        raise ValueError("Expected 0 < --ratio-min <= --ratio-max")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    volumes, total_volume_rows = eligible_volumes(
        args.volumes_parquet,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    print(
        f"Selected {len(volumes)} ratio-compatible volumes from "
        f"{total_volume_rows} volume records",
        flush=True,
    )
    selection_target = args.count + max(10, round(args.count * 0.05))
    selected, selection_stats = select_candidates(
        volumes,
        args=args,
        target_count=selection_target,
    )
    if len(selected) < args.count:
        raise RuntimeError(
            f"Found only {len(selected)} candidates after scanning all completed volumes"
        )
    print(
        f"Selected {len(selected)} candidates (including download fallbacks) across "
        f"{selection_stats['distinct_selected_volumes']} volumes",
        flush=True,
    )
    downloaded, errors = download_selected(selected, args=args)
    if len(downloaded) != args.count:
        raise RuntimeError(
            f"Downloaded {len(downloaded)}/{args.count} pages; see errors in the terminal"
        )
    make_contact_sheets(
        downloaded,
        destination=args.out_dir / "contact_sheets",
        sheet_size=args.contact_sheet_size,
    )
    write_outputs(downloaded, errors, args=args, selection_stats=selection_stats)
    print(f"Wrote review set to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
