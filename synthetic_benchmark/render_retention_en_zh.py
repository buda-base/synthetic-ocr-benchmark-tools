#!/usr/bin/env python3
"""Render English and Chinese OCR-retention pages with the BCS06 raster policy.

Text is fetched from Wikipedia through the MediaWiki API (CC BY-SA 4.0), split
by source document, rendered with Pillow/RAQM, and passed through the same
document augmentation implementation used by BECSynthetic_06.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from document_augmentation import (
    apply_document_augmentations,
    assign_document_augmentations,
    enforce_background_readability_for_line_count,
    write_document_augmentation_manifest,
)
from paper_backgrounds import (
    load_paper_backgrounds,
    prepare_paper_background_cache,
)


BENCHMARK_VERSION = "202604"
VOLUME_VERSION = "v001"
GROUP_SIZE = 1000
SEED = 260417
WIKIPEDIA_LICENSE = "CC BY-SA 4.0; https://creativecommons.org/licenses/by-sa/4.0/"
DATASETS = {
    "en": {
        "dataset_id": "BECSynthetic_RetainEN",
        "mw_id": "MW1BCS007",
        "w_id": "W1BCS007",
        "ie_id": "IE1BCS007",
        "i_prefix": "I1BCS007",
        "ve_prefix": "VE1BCS007",
    },
    "zh": {
        "dataset_id": "BECSynthetic_RetainZH",
        "mw_id": "MW1BCS008",
        "w_id": "W1BCS008",
        "ie_id": "IE1BCS008",
        "i_prefix": "I1BCS008",
        "ve_prefix": "VE1BCS008",
    },
}

# Faces, rather than broad families, are split-disjoint. Fontconfig's selected
# family/style and TTC index are retained in every alignment row.
FONT_SPECS = {
    "en": {
        "train": [
            ("Noto Serif:style=Regular", "SIL Open Font License 1.1"),
            ("Noto Sans:style=Regular", "SIL Open Font License 1.1"),
            ("Liberation Serif:style=Regular", "SIL Open Font License 1.1"),
            ("Liberation Sans:style=Regular", "SIL Open Font License 1.1"),
            ("DejaVu Serif:style=Book", "Bitstream Vera license"),
            ("DejaVu Sans:style=Book", "Bitstream Vera license"),
            ("DejaVu Sans Mono:style=Book", "Bitstream Vera license"),
            ("EB Garamond:style=Regular", "SIL Open Font License 1.1"),
        ],
        "val": [
            ("Noto Serif:style=Bold", "SIL Open Font License 1.1"),
            ("Noto Sans:style=Bold", "SIL Open Font License 1.1"),
            ("Liberation Mono:style=Regular", "SIL Open Font License 1.1"),
        ],
        "test": [
            ("Noto Serif:style=Italic", "SIL Open Font License 1.1"),
            ("Noto Sans:style=Italic", "SIL Open Font License 1.1"),
            ("Liberation Serif:style=Italic", "SIL Open Font License 1.1"),
        ],
    },
    "zh": {
        "train": [
            ("Noto Serif CJK SC:style=Regular", "SIL Open Font License 1.1"),
            ("Noto Sans CJK SC:style=Regular", "SIL Open Font License 1.1"),
            ("WenQuanYi Zen Hei:style=Regular", "GNU GPL v2 with font embedding exception"),
            ("WenQuanYi Micro Hei:style=Regular", "Apache License 2.0 / GPLv3 dual license"),
            ("AR PL UKai CN:style=Book", "Arphic Public License"),
            ("AR PL UMing CN:style=Light", "Arphic Public License"),
        ],
        "val": [
            ("Noto Serif CJK SC:style=Bold", "SIL Open Font License 1.1"),
            ("Noto Sans CJK SC:style=Bold", "SIL Open Font License 1.1"),
            ("Noto Sans CJK TC:style=Regular", "SIL Open Font License 1.1"),
        ],
        "test": [
            ("Noto Serif CJK SC:style=SemiBold", "SIL Open Font License 1.1"),
            ("Noto Sans CJK SC:style=Medium", "SIL Open Font License 1.1"),
            ("Noto Serif CJK TC:style=Regular", "SIL Open Font License 1.1"),
        ],
    },
}


@dataclass(frozen=True)
class FontFace:
    requested: str
    path: str
    family: str
    style: str
    index: int
    license: str

    @property
    def name(self) -> str:
        return f"{self.family} {self.style}".strip()


def stable_int(*values: object) -> int:
    payload = "\0".join(str(v) for v in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def normalize_target(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return unicodedata.normalize("NFC", "\n".join(lines).rstrip())


def fc_match(spec: str, license_name: str) -> FontFace | None:
    result = subprocess.run(
        ["fc-match", "-f", "%{file}\t%{family}\t%{style}\t%{index}\n", spec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not result:
        return None
    fields = result[0].split("\t")
    if len(fields) < 4 or not Path(fields[0]).exists():
        return None
    requested_family = spec.split(":", 1)[0].lower()
    if requested_family not in fields[1].lower():
        return None
    return FontFace(spec, fields[0], fields[1].split(",")[0], fields[2].split(",")[0], int(fields[3] or 0), license_name)


def discover_fonts() -> dict[str, dict[str, list[FontFace]]]:
    result: dict[str, dict[str, list[FontFace]]] = {}
    for language, splits in FONT_SPECS.items():
        result[language] = {}
        for split, specs in splits.items():
            faces = [face for spec, lic in specs if (face := fc_match(spec, lic))]
            if len(faces) < 2:
                raise RuntimeError(f"Too few {language}/{split} fonts installed: {faces}")
            result[language][split] = faces
    names_by_split = {
        split: {face.name for lang in result.values() for face in lang[split]}
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = names_by_split[left] & names_by_split[right]
        if overlap:
            raise RuntimeError(f"Font faces span {left}/{right}: {sorted(overlap)}")
    return result


def api_json(language: str, params: dict[str, object]) -> dict:
    query = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "origin": "*",
        **params,
    }
    url = f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "BDRC-Synthetic-OCR/1.0 (https://www.bdrc.io/)"}
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 7:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 5.0 * 2**attempt)
            time.sleep(delay)
    else:
        raise RuntimeError("MediaWiki request retry loop exhausted")
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API error: {payload['error']}")
    return payload


def clean_extract(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def fetch_wikipedia_documents(
    language: str,
    *,
    target: int,
    variant: str | None,
    cache_path: Path,
) -> list[dict]:
    docs: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            docs[str(item["pageid"])] = item
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    while len(docs) < target:
        params: dict[str, object] = {
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 20,
            "prop": "info",
            "inprop": "url",
        }
        if variant:
            params["variant"] = variant
        payload = api_json(language, params)
        added = 0
        for page in payload.get("query", {}).get("pages", []):
            # Whole-article extracts are limited to one page per request. Keep
            # this deliberately below Wikimedia's anonymous bot thresholds.
            time.sleep(0.35)
            extract_params: dict[str, object] = {
                "pageids": page["pageid"],
                "prop": "extracts|info",
                "inprop": "url",
                "explaintext": 1,
                "exsectionformat": "plain",
            }
            if variant:
                extract_params["variant"] = variant
            detail = api_json(language, extract_params)
            detail_pages = detail.get("query", {}).get("pages", [])
            if not detail_pages:
                continue
            page = detail_pages[0]
            extract = clean_extract(str(page.get("extract") or ""))
            if len(extract) < 500:
                continue
            item = {
                "pageid": str(page["pageid"]),
                "title": unicodedata.normalize("NFC", str(page["title"])),
                "url": str(page.get("fullurl") or ""),
                "text": extract,
                "source": f"{language}.wikipedia.org",
                "license": WIKIPEDIA_LICENSE,
                "variant": variant or language,
            }
            docs[item["pageid"]] = item
            added += 1
        if not added:
            raise RuntimeError(f"MediaWiki returned no usable {language} documents")
        cache_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in docs.values()),
            encoding="utf-8",
        )
    return list(docs.values())


def doc_split(doc: dict) -> str:
    value = stable_int("split", doc["source"], doc["pageid"]) % 100
    if value < 6:
        return "val"
    if value < 12:
        return "test"
    return "train"


def load_eval_texts(path: Path) -> dict[str, list[str]]:
    table = pq.read_table(path, columns=["transcription"])
    result = {"en": [], "zh": []}
    for value in table.column(0).to_pylist():
        text = normalize_compare(str(value or ""))
        if not text:
            continue
        lang = "zh" if sum("\u3400" <= ch <= "\u9fff" for ch in text) > len(text) * 0.15 else "en"
        result[lang].append(text)
    return result


def normalize_compare(text: str) -> str:
    return "".join(ch.lower() for ch in unicodedata.normalize("NFKC", text) if ch.isalnum())


@lru_cache(maxsize=2048)
def compare_shingles(text: str) -> frozenset[str]:
    return frozenset(text[i : i + 13] for i in range(max(1, len(text) - 12)))


def near_duplicate(
    candidate: str,
    eval_texts: list[str],
    eval_union: frozenset[str] | None = None,
) -> tuple[bool, float]:
    candidate = normalize_compare(candidate)
    if len(candidate) < 20:
        return False, 0.0
    shingles = compare_shingles(candidate)
    if eval_union is not None and not (shingles & eval_union):
        return False, 0.0
    best = 0.0
    for reference in eval_texts:
        ref_shingles = compare_shingles(reference)
        if not (shingles & ref_shingles):
            continue
        if min(len(candidate), len(reference)) >= 40 and (candidate in reference or reference in candidate):
            return True, 1.0
        overlap = len(shingles & ref_shingles)
        score = overlap / max(1, min(len(shingles), len(ref_shingles)))
        best = max(best, score)
        if score >= 0.80:
            return True, score
    return False, best


def split_paragraphs(doc: dict) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", doc["text"]) if len(part.strip()) >= 20]


def source_slice(doc: dict, page_id: int, region_kind: str, full_page: bool) -> str:
    paragraphs = split_paragraphs(doc)
    rng = random.Random(stable_int("source", doc["pageid"], page_id))
    start = rng.randrange(len(paragraphs))
    ordered = paragraphs[start:] + paragraphs[:start]
    if region_kind == "title":
        text = doc["title"] + "\n" + ordered[0]
    elif region_kind == "caption":
        text = ordered[0]
    elif region_kind == "footnote":
        text = ordered[0]
    elif region_kind == "reference":
        text = "\n".join(f"[{i + 1}] {part}" for i, part in enumerate(ordered[:5]))
    else:
        text = "\n\n".join(ordered)
    target_chars = rng.randint(1800, 5000) if full_page else rng.randint(20, 1500)
    if len(text) < target_chars:
        text = (text + "\n\n" + "\n\n".join(ordered))[:target_chars]
    else:
        text = text[:target_chars]
    return text.strip()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return float(draw.textlength(text, font=font, direction="ltr"))


def wrap_english(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if current and text_width(draw, candidate, font) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def wrap_chinese(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    no_line_start = set("，。！？；：、）》」』】〕〉…")
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and text_width(draw, candidate, font) > width:
                if char in no_line_start:
                    current += char
                    continue
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def make_plan(
    docs: dict[str, list[dict]],
    fonts: dict[str, dict[str, list[FontFace]]],
    eval_texts: dict[str, list[str]],
    count_per_language: int,
    *,
    start_image_id: int = 1,
    train_only: bool = False,
) -> tuple[list[dict], dict]:
    pages: list[dict] = []
    dup_report = {"evaluated": 0, "rejected": 0, "maximum_non_rejected_score": 0.0}
    split_counts = (
        {"train": count_per_language, "val": 0, "test": 0}
        if train_only or count_per_language < 600
        else {"train": count_per_language - 600, "val": 300, "test": 300}
    )
    eval_unions = {
        language: frozenset().union(*(compare_shingles(text) for text in texts))
        for language, texts in eval_texts.items()
    }
    region_kinds = ("text", "title", "caption", "footnote", "reference")
    for language in ("en", "zh"):
        language_offset = 0
        by_split = defaultdict(list)
        for doc in docs[language]:
            by_split[doc_split(doc)].append(doc)
        for split in ("train", "val", "test"):
            if split_counts[split] == 0:
                continue
            if not by_split[split]:
                raise RuntimeError(f"No {language} source documents assigned to {split}")
            for local_index in range(split_counts[split]):
                image_id = start_image_id + language_offset
                language_offset += 1
                full_page = image_id % 2 == 0
                region_kind = "page" if full_page else region_kinds[(image_id // 2) % len(region_kinds)]
                candidates = by_split[split]
                attempt = 0
                while True:
                    doc = candidates[(stable_int(language, split, image_id, attempt) + local_index) % len(candidates)]
                    text = source_slice(doc, image_id + attempt, region_kind, full_page)
                    is_dup, score = near_duplicate(
                        text, eval_texts[language], eval_unions[language]
                    )
                    dup_report["evaluated"] += 1
                    if not is_dup:
                        dup_report["maximum_non_rejected_score"] = max(
                            dup_report["maximum_non_rejected_score"], score
                        )
                        break
                    dup_report["rejected"] += 1
                    attempt += 1
                    if attempt > len(candidates) * 2:
                        raise RuntimeError(f"Could not find non-duplicate {language}/{split} source")
                face_pool = fonts[language][split]
                face = face_pool[stable_int("font", language, image_id) % len(face_pool)]
                zh_hant = language == "zh" and doc.get("variant") == "zh-hant"
                pages.append(
                    {
                        "image_id": image_id,
                        "language_group": language,
                        "language": "zh-Hant" if zh_hant else ("zh-Hans" if language == "zh" else "en"),
                        "suggested_split": split,
                        "layout_class": "page" if full_page else "region",
                        "layout_kind": ("stack" if image_id % 4 == 0 else "single") if full_page else "crop",
                        "region_kind": region_kind,
                        "source_text": text,
                        "source_document_id": f"{doc['source']}:{doc['pageid']}",
                        "etext_source": f"{doc['url']} | {doc['license']}",
                        "source_title": doc["title"],
                        "font": asdict(face),
                        "basename": face.name,
                        "ttc_face_index": str(face.index),
                        "font_size_pt": 24.0,
                        "layout_font_scale_multiplier": 1.0,
                        "text_length_bucket": "short" if not full_page else "normal",
                        "output_width_px": 1800 + stable_int("width", language, image_id) % 1701,
                        "output_jpeg_quality": 65 if stable_int("jpeg", language, image_id) % 10 == 0 else 85,
                    }
                )
    pages.sort(key=lambda row: (row["language_group"], int(row["image_id"])))
    return pages, dup_report


def image_ids(row: dict) -> tuple[str, str, int]:
    volume = (int(row["image_id"]) - 1) // GROUP_SIZE + 1
    page = (int(row["image_id"]) - 1) % GROUP_SIZE + 1
    ids = DATASETS[row["language_group"]]
    return f"{ids['i_prefix']}_{volume:04d}", f"{ids['ve_prefix']}_{volume:04d}", page


def render_one(payload: tuple[dict, str]) -> dict:
    row, out_dir_string = payload
    out_dir = Path(out_dir_string)
    rng = random.Random(stable_int("render", row["language_group"], row["image_id"]))
    width = int(row["output_width_px"])
    face = row["font"]
    font_px = rng.randint(30, 70)
    font = ImageFont.truetype(
        face["path"],
        font_px,
        index=int(face["index"]),
        layout_engine=ImageFont.Layout.RAQM,
    )
    scratch = Image.new("L", (width, 100), 255)
    draw = ImageDraw.Draw(scratch)
    if row["layout_class"] == "region":
        margin_x = max(8, int(width * rng.uniform(0.006, 0.025)))
        max_text_width = width - 2 * margin_x
        max_lines = rng.randint(1, 30)
        aspect = None
    else:
        aspect = math.exp(rng.uniform(math.log(0.35), math.log(2.2)))
        height = min(3500, max(700, round(width / aspect)))
        margin_x = int(min(width, height) * rng.uniform(0.01, 0.15))
        max_text_width = max(100, width - 2 * margin_x)
        max_lines = 60
    wrap = wrap_english if row["language_group"] == "en" else wrap_chinese
    lines = wrap(row["source_text"], draw, font, max_text_width)
    line_gap = round(font_px * rng.uniform(0.05, 0.60))
    pitch = font_px + line_gap
    if row["layout_class"] == "page":
        margin_y = int(min(width, height) * rng.uniform(0.005, 0.15))
        max_lines = max(1, min(max_lines, (height - 2 * margin_y) // pitch))
    else:
        # A sampled one-line crop can otherwise stop at a short section heading.
        # Extend it only as far as needed to satisfy the 20-character region
        # floor, while retaining the requested hard ceiling of 30 visual lines.
        selected = lines[:max_lines]
        cursor = len(selected)
        while sum(len(line) for line in selected) < 20 and cursor < min(30, len(lines)):
            selected.append(lines[cursor])
            cursor += 1
        lines = selected
        if sum(len(line) for line in lines) < 20:
            expanded = wrap(
                row["source_text"] + " " + row["source_title"],
                draw,
                font,
                max_text_width,
            )
            lines = expanded[: max(max_lines, min(30, len(expanded)))]
        margin_y = max(6, round(font_px * rng.uniform(0.20, 0.75)))
        height = max(2 * margin_y + pitch, 2 * margin_y + len(lines) * pitch)
    lines = lines[: (max_lines if row["layout_class"] == "page" else 30)]
    while lines and lines[-1] == "":
        lines.pop()
    if row["layout_kind"] == "stack" and len(lines) >= 8:
        gap_at = len(lines) // 2
        lines.insert(gap_at, "")
        lines = lines[:max_lines]
    if not lines or not any(lines):
        raise RuntimeError(f"Empty rendered label for {row['language_group']} {row['image_id']}")
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    y = margin_y
    for line in lines:
        if line:
            draw.text((margin_x, y), line, fill=0, font=font, direction="ltr")
        y += pitch
    target = normalize_target("\n".join(lines))
    i_id, ve_id, page = image_ids(row)
    extension = str(row.get("document_augmentation_output_extension") or ".jpg")
    relative = Path("images") / DATASETS[row["language_group"]]["w_id"] / i_id / VOLUME_VERSION / f"{page:04d}{extension}"
    destination = out_dir / relative
    raw = out_dir / "raw" / row["language_group"] / f"{int(row['image_id']):05d}.png"
    raw.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(raw)
    enforce_background_readability_for_line_count(row, len(lines))
    apply_document_augmentations(raw, destination, row, jpeg_quality=int(row["output_jpeg_quality"]))
    raw.unlink(missing_ok=True)
    result = {k: v for k, v in row.items() if not k.startswith("_")}
    result.update(
        {
            "transcription": target,
            "font_name": FontFace(**face).name,
            "font_license": face["license"],
            "font_file": face["path"],
            "font_style": face["style"],
            "effective_em_px": font_px,
            "line_spacing_px": pitch,
            "line_count": len(lines),
            "character_count": len(target.replace("\n", "")),
            "layout_recipe_kind": row["layout_kind"],
            "layout_recipe_aspect_sampled": aspect,
            "layout_recipe_aspect_realized": width / height,
            "layout_recipe_block_count": 2 if row["layout_kind"] == "stack" else 1,
            "layout_recipe_realized_line_count": len(lines),
            "layout_recipe_effective_em_px": font_px,
            "i_id": i_id,
            "ve_id": ve_id,
            "page_in_volume": page,
            "image_relpath": str(relative),
            "img_file_name": destination.name,
        }
    )
    result.pop("font", None)
    result.pop("source_text", None)
    return result


def alignment_row(row: dict) -> dict:
    excluded = {
        "image_id", "language_group", "i_id", "ve_id", "page_in_volume",
        "image_relpath", "layout_class", "layout_kind",
    }
    return parquet_row({key: value for key, value in row.items() if key not in excluded})


def parquet_row(row: dict) -> dict:
    """Keep deterministic unsigned-style seeds portable across Arrow int64."""
    nullable_numeric = {
        "document_augmentation_background_mean_luminance",
        "document_augmentation_effective_text_scale",
        "document_augmentation_rotation_deg",
        "document_augmentation_tps_y_norm",
    }
    result = {}
    for key, value in row.items():
        if key.endswith("seed") and value not in (None, ""):
            result[key] = str(value)
        elif key in nullable_numeric and value == "":
            result[key] = None
        else:
            result[key] = value
    return result


def write_outputs(out_dir: Path, rows: list[dict], manifests: dict) -> None:
    align_root = out_dir / "alignments" / BENCHMARK_VERSION
    for language in ("en", "zh"):
        ids = DATASETS[language]
        dataset_root = align_root / ids["dataset_id"]
        dataset_root.mkdir(parents=True, exist_ok=True)
        lang_rows = [row for row in rows if row["language_group"] == language]
        grouped = defaultdict(list)
        for row in lang_rows:
            grouped[(row["i_id"], row["ve_id"])].append(row)
        for (i_id, ve_id), group in sorted(grouped.items()):
            pq.write_table(
                pa.Table.from_pylist([alignment_row(row) for row in sorted(group, key=lambda x: x["page_in_volume"])]),
                dataset_root / f"{i_id}-{ve_id}.parquet",
                compression="zstd",
            )
        with (dataset_root / "catalog_alignments.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["i_id", "ve_id", "alignment_type", "nb_pages", "mode"])
            writer.writeheader()
            for (i_id, ve_id), group in sorted(grouped.items()):
                writer.writerow({"i_id": i_id, "ve_id": ve_id, "alignment_type": "manual", "nb_pages": len(group), "mode": "ptt"})
        with (dataset_root / "catalog_volumes.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["mw_id", "mw_title", "coll_id", "vol_id", "vol_version", "source", "line_breaks", "technology", "script", "access"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for (i_id, ve_id), group in sorted(grouped.items()):
                for coll_id, vol_id in ((ids["w_id"], i_id), (ids["ie_id"], ve_id)):
                    writer.writerow({
                        "mw_id": ids["mw_id"],
                        "mw_title": f"Synthetic {'English' if language == 'en' else 'Chinese'} OCR retention replay",
                        "coll_id": coll_id, "vol_id": vol_id, "vol_version": VOLUME_VERSION,
                        "source": "Wikipedia", "line_breaks": "Y", "technology": "synthetic",
                        "script": "latin" if language == "en" else "han", "access": "open",
                    })
        (dataset_root / "README.md").write_text(
            f"# {ids['dataset_id']}\n\nSynthetic OCR retention pages rendered from Wikipedia "
            f"(CC BY-SA 4.0) with source URLs and font licences in each alignment row. "
            "Targets are NFC and preserve exact rendered line endings.\n",
            encoding="utf-8",
        )
    with (align_root / "datasets.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["dataset_id", "nb_img_volumes", "nb_img_files", "technology", "script", "modes", "line_breaks"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for language in ("en", "zh"):
            lang_rows = [row for row in rows if row["language_group"] == language]
            writer.writerow({
                "dataset_id": DATASETS[language]["dataset_id"],
                "nb_img_volumes": len({row["i_id"] for row in lang_rows}),
                "nb_img_files": len(lang_rows), "technology": "synthetic",
                "script": "latin" if language == "en" else "han", "modes": "ptt", "line_breaks": "Y",
            })
    (out_dir / "document_augmentation_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_contact_sheet(out_dir: Path, rows: list[dict], language: str) -> Path:
    selected = [row for row in rows if row["language_group"] == language]
    selected = [selected[i] for i in [round(j * (len(selected) - 1) / 19) for j in range(20)]]
    thumbs = []
    for row in selected:
        image = Image.open(out_dir / row["image_relpath"]).convert("RGB")
        image.thumbnail((420, 300))
        tile = Image.new("RGB", (440, 340), "white")
        tile.paste(image, ((440 - image.width) // 2, 5))
        ImageDraw.Draw(tile).text((8, 315), f"{row['language']} {row['image_id']} · {row['font_name'][:38]}", fill="black")
        thumbs.append(tile)
    sheet = Image.new("RGB", (440 * 4, 340 * 5), (220, 220, 220))
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % 4) * 440, (i // 4) * 340))
    path = out_dir / "reports" / f"contact_sheet_{language}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)
    return path


def write_report(out_dir: Path, rows: list[dict], fonts: dict, dup_report: dict) -> None:
    counts = {
        "language": dict(Counter(row["language"] for row in rows)),
        "layout_class": dict(Counter(row["layout_class"] for row in rows)),
        "layout_kind": dict(Counter(row["layout_recipe_kind"] for row in rows)),
        "split": dict(Counter(f"{row['language_group']}:{row['suggested_split']}" for row in rows)),
        "region_kind": dict(Counter(row["region_kind"] for row in rows)),
    }
    chars = [int(row["character_count"]) for row in rows]
    lines = [int(row["line_count"]) for row in rows]
    report = {
        "counts": counts,
        "characters_per_page": {"median": median(chars), "p90": quantile(chars, 0.9)},
        "lines_per_page": {"median": median(lines), "p90": quantile(lines, 0.9)},
        "fonts": sorted(
            {f"{row['font_name']} | {row['font_license']} | {row['font_file']}" for row in rows}
        ),
        "sources": {
            "English Wikipedia": WIKIPEDIA_LICENSE,
            "Chinese Wikipedia": WIKIPEDIA_LICENSE,
        },
        "near_duplicate_check": dup_report,
        "datasets": DATASETS,
        "image_s3_roots": {
            language: f"s3://bec.bdrc.io/ocr_benchmark/images/{ids['w_id']}/"
            for language, ids in DATASETS.items()
        },
        "alignment_s3_roots": {
            language: f"s3://bec.bdrc.io/ocr_benchmark/alignments/{BENCHMARK_VERSION}/{ids['dataset_id']}/"
            for language, ids in DATASETS.items()
        },
    }
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines_md = [
        "# English/Chinese OCR retention render",
        "",
        f"- Counts: `{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`",
        f"- Characters/page: median {median(chars):.0f}, p90 {quantile(chars, 0.9):.0f}",
        f"- Lines/page: median {median(lines):.0f}, p90 {quantile(lines, 0.9):.0f}",
        f"- Near-duplicate rejections: {dup_report['rejected']} of {dup_report['evaluated']} candidates",
        "",
        "## Fonts",
        *[f"- {item}" for item in report["fonts"]],
        "",
        "## Sources",
        f"- English Wikipedia — {WIKIPEDIA_LICENSE}",
        f"- Chinese Wikipedia — {WIKIPEDIA_LICENSE}",
    ]
    (reports / "README.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-parquet", type=Path, required=True)
    parser.add_argument("--count-per-language", type=int, default=5000)
    parser.add_argument("--start-image-id", type=int, default=1)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--source-docs-per-variant", type=int, default=500)
    parser.add_argument(
        "--prefer-new-source-manifest",
        type=Path,
        default=None,
        help="Use source documents absent from this earlier render manifest when possible.",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--cache-dir", type=Path, default=Path("/mnt/synthbench/retention_cache"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count_per_language <= 0:
        raise SystemExit("--count-per-language must be positive")
    if args.start_image_id <= 0:
        raise SystemExit("--start-image-id must be positive")
    fonts = discover_fonts()
    en_docs = fetch_wikipedia_documents(
        "en", target=args.source_docs_per_variant, variant=None,
        cache_path=args.cache_dir / "wikipedia_en.jsonl",
    )
    hans_docs = fetch_wikipedia_documents(
        "zh", target=args.source_docs_per_variant, variant="zh-hans",
        cache_path=args.cache_dir / "wikipedia_zh_hans.jsonl",
    )
    hant_docs = fetch_wikipedia_documents(
        "zh", target=max(10, args.source_docs_per_variant // 9), variant="zh-hant",
        cache_path=args.cache_dir / "wikipedia_zh_hant.jsonl",
    )
    if args.prefer_new_source_manifest:
        previous = pq.read_table(
            args.prefer_new_source_manifest,
            columns=["language_group", "source_document_id"],
        ).to_pylist()
        used = defaultdict(set)
        for row in previous:
            used[str(row["language_group"])].add(str(row["source_document_id"]))

        def new_documents(items: list[dict], language: str) -> list[dict]:
            fresh = [
                doc
                for doc in items
                if f"{doc['source']}:{doc['pageid']}" not in used[language]
            ]
            return fresh or items

        en_docs = new_documents(en_docs, "en")
        hans_docs = new_documents(hans_docs, "zh")
        hant_docs = new_documents(hant_docs, "zh")

    # Keep the Chinese source pool at the Batch 1 ratio (about 90/10).
    ratio_units = min(len(hans_docs) // 9, len(hant_docs))
    if ratio_units:
        zh_docs = hans_docs[: 9 * ratio_units] + hant_docs[:ratio_units]
    else:
        zh_docs = hans_docs + hant_docs
    eval_texts = load_eval_texts(args.eval_parquet)
    rows, dup_report = make_plan(
        {"en": en_docs, "zh": zh_docs},
        fonts,
        eval_texts,
        args.count_per_language,
        start_image_id=args.start_image_id,
        train_only=args.train_only,
    )
    backgrounds = load_paper_backgrounds()
    rows, augmentation_manifest = assign_document_augmentations(rows, paper_backgrounds=backgrounds, seed=SEED)
    prepare_paper_background_cache(
        rows, backgrounds, cache_dir=args.cache_dir / "paper_backgrounds", workers=min(16, args.jobs * 2)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "render_plan.parquet"
    pq.write_table(
        pa.Table.from_pylist([
            parquet_row({k: v for k, v in row.items() if not k.startswith("_")})
            for row in rows
        ]),
        plan_path,
        compression="zstd",
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        results = list(
            tqdm(
                executor.map(render_one, ((row, str(args.out_dir)) for row in rows), chunksize=8),
                total=len(rows), desc="Render retention pages", unit="page",
            )
        )
    results.sort(key=lambda row: (row["language_group"], int(row["image_id"])))
    write_outputs(args.out_dir, results, augmentation_manifest)
    write_report(args.out_dir, results, fonts, dup_report)
    make_contact_sheet(args.out_dir, results, "en")
    make_contact_sheet(args.out_dir, results, "zh")
    pq.write_table(
        pa.Table.from_pylist([parquet_row(row) for row in results]),
        args.out_dir / "render_manifest.parquet",
        compression="zstd",
    )


if __name__ == "__main__":
    main()
