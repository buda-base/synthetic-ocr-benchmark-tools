#!/usr/bin/env python3
"""Download a small per-font image sample from an uploaded synthetic benchmark."""

from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from synthetic_common import DEFAULT_OUTPUT_DIR

W_ID = "W1BCS001"
I_ID_PREFIX = "I1BCS001"
VE_ID_PREFIX = "VE1BCS001"
VOLUME_VERSION = "v001"
BENCHMARK_VERSION = "202604"
DATASET_ID = "BECSynthetic_01"
GROUP_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download N benchmark images per font from S3.")
    parser.add_argument("--s3-root", default="s3://bec.bdrc.io/ocr_benchmark")
    parser.add_argument("--samples-per-font", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/synthetic_benchmark/out/s3_font_samples"))
    parser.add_argument(
        "--checkpoint-cache",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "s3_checkpoint_cache" / "catalog_batches",
        help="Local cache for S3 checkpoint CSV fragments.",
    )
    parser.add_argument("--render-plan", type=Path, default=Path("scripts/synthetic_benchmark/out/render_plan.parquet"))
    parser.add_argument(
        "--alignment-cache",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "s3_alignment_cache" / DATASET_ID,
        help="Local cache for S3 alignment parquet files.",
    )
    parser.add_argument("--aws", default="aws", help="AWS CLI executable")
    return parser.parse_args()


def s3_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts if part)])


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_optional(command: list[str]) -> bool:
    return (
        subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def print_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def sync_checkpoint_fragments(args: argparse.Namespace) -> None:
    args.checkpoint_cache.mkdir(parents=True, exist_ok=True)
    run(
        [
            args.aws,
            "s3",
            "sync",
            s3_join(args.s3_root, "checkpoints/catalog_batches"),
            str(args.checkpoint_cache),
            "--only-show-errors",
        ]
    )


def read_checkpoint_rows(checkpoint_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fragment_path in sorted(checkpoint_dir.glob("*.csv")):
        with fragment_path.open(encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def benchmark_image_relpath(output_id: int, extension: str = ".jpg") -> str:
    volume_no = ((output_id - 1) // GROUP_SIZE) + 1
    page_no = ((output_id - 1) % GROUP_SIZE) + 1
    return (
        f"images/{W_ID}/{I_ID_PREFIX}_{volume_no:04d}/"
        f"{VOLUME_VERSION}/{page_no:04d}{extension}"
    )


def volume_number(output_id: int) -> int:
    return ((output_id - 1) // GROUP_SIZE) + 1


def i_id_for_output(output_id: int) -> str:
    return f"{I_ID_PREFIX}_{volume_number(output_id):04d}"


def ve_id_for_output(output_id: int) -> str:
    return f"{VE_ID_PREFIX}_{volume_number(output_id):04d}"


def i_id_from_image_path(image_file_name: str) -> str:
    parts = Path(image_file_name).parts
    for part in parts:
        if part.startswith(f"{I_ID_PREFIX}_"):
            return part
    output_id = int(Path(image_file_name).stem)
    return i_id_for_output(output_id)


def read_plan_rows(render_plan: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in pq.read_table(render_plan).to_pylist():
        image_id = int(row["image_id"])
        rows.append(
            {
                "output_sequence": str(image_id),
                "image_id": str(image_id),
                "image_file_name": benchmark_image_relpath(image_id),
                "source_plan_image_ids": str(image_id),
                "chunk_id": str(row.get("chunk_id") or ""),
                "ps_name": str(row.get("ps_name") or ""),
                "font_file": str(row.get("font_file") or ""),
                "font_path": str(row.get("font_path") or ""),
            }
        )
    return rows


def download_alignment_catalog(args: argparse.Namespace) -> Path | None:
    local_path = args.alignment_cache / "catalog_alignments.csv"
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    ok = run_optional(
        [
            args.aws,
            "s3",
            "cp",
            s3_join(args.s3_root, "alignments", BENCHMARK_VERSION, DATASET_ID, "catalog_alignments.csv"),
            str(local_path),
            "--only-show-errors",
        ]
    )
    return local_path if ok and local_path.exists() else None


def filter_rows_to_alignment_catalog(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    catalog_path = download_alignment_catalog(args)
    if catalog_path is None:
        print("WARNING: could not download alignment catalog; sampling from the full local plan.")
        return rows

    uploaded_ranges: list[tuple[int, int]] = []
    with catalog_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            i_id = str(row.get("i_id") or "")
            if not i_id.startswith(f"{I_ID_PREFIX}_"):
                continue
            try:
                volume_no = int(i_id.rsplit("_", 1)[-1])
                nb_pages = int(row.get("nb_pages") or "0")
            except ValueError:
                continue
            if nb_pages > 0:
                start = (volume_no - 1) * GROUP_SIZE + 1
                uploaded_ranges.append((start, start + nb_pages - 1))

    if not uploaded_ranges:
        print("WARNING: alignment catalog had no usable volume/page ranges; sampling from the full local plan.")
        return rows

    uploaded_ranges.sort()
    print(f"Using {len(uploaded_ranges)} uploaded volume range(s) from catalog_alignments.csv")

    def is_uploaded(output_sequence: int) -> bool:
        # The catalog has one small range per volume, so a linear scan is fine.
        return any(start <= output_sequence <= end for start, end in uploaded_ranges)

    return [row for row in rows if is_uploaded(row_sort_key(row))]


def ensure_alignment_parquet(args: argparse.Namespace, i_id: str) -> Path | None:
    volume = i_id.rsplit("_", 1)[-1]
    ve_id = f"{VE_ID_PREFIX}_{volume}"
    local_path = args.alignment_cache / f"{i_id}-{ve_id}_ptt.parquet"
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    ok = run_optional(
        [
            args.aws,
            "s3",
            "cp",
            s3_join(
                args.s3_root,
                "alignments",
                BENCHMARK_VERSION,
                DATASET_ID,
                f"{i_id}-{ve_id}_ptt.parquet",
            ),
            str(local_path),
            "--only-show-errors",
        ]
    )
    return local_path if ok and local_path.exists() else None


def load_alignment_volume(args: argparse.Namespace, i_id: str) -> dict[str, str]:
    parquet_path = ensure_alignment_parquet(args, i_id)
    if parquet_path is None:
        return {}
    return {
        str(item["img_file_name"]): str(item.get("transcription") or "")
        for item in pq.read_table(parquet_path, columns=["img_file_name", "transcription"]).to_pylist()
    }


def get_transcription(
    args: argparse.Namespace,
    cache: dict[str, dict[str, str]],
    image_file_name: str,
) -> str | None:
    i_id = i_id_from_image_path(image_file_name)
    if i_id not in cache:
        cache[i_id] = load_alignment_volume(args, i_id)
    return cache[i_id].get(Path(image_file_name).name)


def backfill_existing_text_files(args: argparse.Namespace) -> int:
    """Write missing .txt files for already-downloaded sample images."""
    alignment_cache: dict[str, dict[str, str]] = {}
    written = 0
    image_paths = sorted(
        [*args.out_dir.glob("**/*.jpg"), *args.out_dir.glob("**/*.tif")]
    )
    for image_path in tqdm(image_paths, desc="Backfill existing text files", unit="image"):
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            continue
        try:
            output_sequence = int(image_path.name.split("_", 1)[0])
        except ValueError:
            print_stderr(
                f"ERROR: image has no TXT and cannot infer output sequence: {image_path}"
            )
            continue
        image_file_name = benchmark_image_relpath(output_sequence, image_path.suffix.lower())
        transcription = get_transcription(args, alignment_cache, image_file_name)
        if transcription is None:
            # Keep the review folder invariant: every visible image should have text.
            print_stderr(
                "ERROR: removing image with no TXT because transcription was not found "
                f"in alignment parquet: local={image_path} benchmark_image={image_file_name}"
            )
            image_path.unlink(missing_ok=True)
            continue
        try:
            txt_path.write_text(transcription, encoding="utf-8")
        except OSError as exc:
            print_stderr(
                "ERROR: image has no TXT because writing the text file failed: "
                f"local={image_path} txt={txt_path} reason={exc}"
            )
            continue
        written += 1
    return written


def font_label(row: dict[str, str]) -> str:
    return row.get("ps_name") or row.get("font_file") or row.get("font_path") or "unknown"


def row_sort_key(row: dict[str, str]) -> int:
    try:
        return int(row.get("output_sequence") or row.get("image_id") or "0")
    except ValueError:
        return 0


def sample_rows_by_font(
    rows: list[dict[str, str]],
    samples_per_font: int,
    seed: int,
    oversample: int = 1,
) -> list[dict[str, str]]:
    by_font: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        image_file_name = (row.get("image_file_name") or "").strip()
        if image_file_name:
            by_font[font_label(row)][image_file_name] = row

    rng = random.Random(seed)
    sampled: list[dict[str, str]] = []
    for font in sorted(by_font):
        candidates = sorted(by_font[font].values(), key=row_sort_key)
        target = samples_per_font * oversample
        if len(candidates) > target:
            candidates = rng.sample(candidates, target)
            candidates.sort(key=row_sort_key)
        sampled.extend(candidates)
    return sampled


def download_samples(args: argparse.Namespace, rows: list[dict[str, str]]) -> Path:
    manifest_path = args.out_dir / "manifest.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written_by_font: dict[str, int] = defaultdict(int)
    alignment_cache: dict[str, dict[str, str]] = {}
    rows_by_font: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_font[font_label(row)].append(row)
    target_count = len(rows_by_font) * args.samples_per_font
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "font",
            "output_sequence",
            "image_file_name",
            "local_file",
            "local_text_file",
            "source_plan_image_ids",
            "chunk_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with tqdm(total=target_count, desc="Download image/text samples", unit="image") as progress:
            for font in sorted(rows_by_font):
                for row in rows_by_font[font]:
                    if written_by_font[font] >= args.samples_per_font:
                        break
                    image_file_name = row["image_file_name"]
                    local_dir = args.out_dir / safe_name(font)
                    local_name = f"{int(row_sort_key(row)):06d}_{Path(image_file_name).name}"
                    local_path = local_dir / local_name
                    local_text_path = local_path.with_suffix(".txt")
                    tmp_image_path = local_path.with_suffix(local_path.suffix + ".tmp")
                    local_dir.mkdir(parents=True, exist_ok=True)

                    transcription = get_transcription(args, alignment_cache, image_file_name)
                    if transcription is None:
                        if local_path.exists():
                            print_stderr(
                                "ERROR: removing JPG with no TXT because transcription was not found "
                                f"in alignment parquet: local={local_path} benchmark_image={image_file_name}"
                            )
                        local_path.unlink(missing_ok=True)
                        tmp_image_path.unlink(missing_ok=True)
                        continue

                    if not local_path.exists():
                        tmp_image_path.unlink(missing_ok=True)
                        ok = run_optional(
                            [
                                args.aws,
                                "s3",
                                "cp",
                                s3_join(args.s3_root, image_file_name),
                                str(tmp_image_path),
                                "--only-show-errors",
                            ]
                        )
                        if not ok:
                            tmp_image_path.unlink(missing_ok=True)
                            continue
                        try:
                            local_text_path.write_text(transcription, encoding="utf-8")
                        except OSError as exc:
                            tmp_image_path.unlink(missing_ok=True)
                            print_stderr(
                                "ERROR: downloaded temporary JPG but did not publish it because "
                                f"writing TXT failed: tmp={tmp_image_path} txt={local_text_path} reason={exc}"
                            )
                            continue
                        tmp_image_path.replace(local_path)
                    else:
                        try:
                            local_text_path.write_text(transcription, encoding="utf-8")
                        except OSError as exc:
                            print_stderr(
                                "ERROR: existing JPG has no TXT because writing the text file failed: "
                                f"local={local_path} txt={local_text_path} reason={exc}"
                            )
                            continue
                    if local_path.exists() and not local_text_path.exists():
                        print_stderr(
                            "ERROR: invariant violation: JPG exists without TXT after sample handling: "
                            f"local={local_path} txt={local_text_path} benchmark_image={image_file_name}"
                        )
                        continue
                    written_by_font[font] += 1
                    progress.update(1)
                    writer.writerow(
                        {
                            "font": font,
                            "output_sequence": row.get("output_sequence") or "",
                            "image_file_name": image_file_name,
                            "local_file": local_path.as_posix(),
                            "local_text_file": local_text_path.as_posix(),
                            "source_plan_image_ids": row.get("source_plan_image_ids") or "",
                            "chunk_id": row.get("chunk_id") or "",
                        }
                    )
    return manifest_path


def main() -> None:
    args = parse_args()
    sync_checkpoint_fragments(args)
    backfilled = backfill_existing_text_files(args)
    if backfilled:
        print(f"Wrote {backfilled} missing text file(s) for existing downloaded images")
    rows = read_checkpoint_rows(args.checkpoint_cache)
    if not rows:
        print(
            f"No checkpoint CSV rows found in {args.checkpoint_cache}; "
            "falling back to render-plan image_id paths."
        )
        print("NOTE: this fallback assumes uploaded image numbering follows the local render plan.")
        rows = read_plan_rows(args.render_plan)
        rows = filter_rows_to_alignment_catalog(args, rows)
        oversample = 10
    else:
        oversample = 1
    sampled_rows = sample_rows_by_font(rows, args.samples_per_font, args.seed, oversample=oversample)
    manifest_path = download_samples(args, sampled_rows)
    with manifest_path.open(encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    fonts = {row["font"] for row in manifest_rows}
    print(f"Downloaded/verified {len(manifest_rows)} image(s) for {len(fonts)} font(s)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
