#!/usr/bin/env python3
"""Render the most common negative-flagged stacks for Kyuma fonts for false-negative review."""

from __future__ import annotations

import argparse
import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from coverage_common import (
    DEFAULT_FONTS_CSV,
    DEFAULT_OUT_DIR,
    hb_view_render,
    load_font_rows,
    slugify,
)

try:  # pyewts is optional; fall back to codepoints when unavailable.
    import pyewts

    @lru_cache(maxsize=1)
    def _ewts() -> "pyewts.pyewts":
        return pyewts.pyewts()

    def to_ewts(text: str) -> str:
        try:
            return _ewts().toWylie(text or "")
        except Exception:  # pragma: no cover - defensive
            return ""

except Exception:  # pragma: no cover - optional dependency
    def to_ewts(text: str) -> str:  # type: ignore[misc]
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("support_parquet", type=Path)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "kyuma_neg_audit")
    parser.add_argument(
        "--kyuma-basenames",
        type=Path,
        required=True,
        help="Newline-delimited Kyuma font basenames (eligible faces).",
    )
    parser.add_argument("--top-per-font", type=int, default=40)
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help="Only render negative stacks seen at least this many times in the corpus.",
    )
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Restrict to is_standard_tibetan==1 (skip Sanskrit-only skt0 rejects).",
    )
    parser.add_argument(
        "--require-reason",
        default=None,
        help="Keep only rows whose ';'-joined reason set contains this token.",
    )
    parser.add_argument(
        "--reasons-subset",
        default=None,
        help=(
            "Comma-separated tokens; keep only rows whose entire reason set is a "
            "subset of these (i.e. the stacks that would flip to OK if these rules "
            "were relaxed). Useful for false-positive review."
        ),
    )
    parser.add_argument("--title-prefix", default="Kyuma negatives")
    parser.add_argument("--hb-view", default="hb-view")
    parser.add_argument("--render-margin", type=int, default=60)
    parser.add_argument("--columns", type=int, default=5)
    return parser.parse_args()


def make_contact_sheet(
    manifest: list[dict[str, str]],
    out_dir: Path,
    *,
    title: str,
    columns: int,
    filename: str,
) -> Path | None:
    if not manifest:
        return None
    cell_w, cell_h = 430, 360
    image_h = 230
    text_y = image_h + 16
    header_h = 40
    rows = (len(manifest) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black")
    for i, item in enumerate(manifest):
        x = (i % columns) * cell_w
        y = header_h + (i // columns) * cell_h
        png_path = out_dir / item["image"]
        if png_path.exists():
            with Image.open(png_path) as img:
                img = img.convert("RGB")
                img.thumbnail((cell_w - 20, image_h))
                sheet.paste(img, (x + 10, y + 10))
        draw.rectangle(
            (x + 2, y + 2, x + cell_w - 3, y + cell_h - 3), outline="red", width=2
        )
        label = (
            f"{i + 1}. occ={item['stack_nb_occurrences']}\n"
            f"{item['label_text']}\n"
            f"reason={item['reason']}"
        )
        draw.multiline_text((x + 10, y + text_y), label, fill="black", spacing=4)
    out_path = out_dir / filename
    sheet.save(out_path, quality=90)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "png"
    image_dir.mkdir(parents=True, exist_ok=True)

    kyuma = [
        line.strip()
        for line in args.kyuma_basenames.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kyuma_set = set(kyuma)

    df = pd.read_parquet(args.support_parquet)
    df = df[(df["test_kind"] == "stack") & (df["basename"].isin(kyuma_set))]
    df = df[df["ok"] == False]  # noqa: E712 - hard negatives only
    if args.standard_only:
        df = df[df["is_standard_tibetan"] == 1]
    # Per-row occurrence is often 0; use the global max per stack as the weight.
    occ_map = (
        pd.read_parquet(args.support_parquet, columns=["stack", "stack_nb_occurrences"])
        .groupby("stack")["stack_nb_occurrences"].max()
    )
    df = df.copy()
    df["stack_nb_occurrences"] = df["stack"].map(occ_map).fillna(0).astype(int)
    if args.require_reason:
        df = df[df["reason"].fillna("").map(
            lambda r: args.require_reason in set(r.split(";")))]
    if args.reasons_subset:
        allowed = {t for t in args.reasons_subset.split(",") if t}
        df = df[df["reason"].fillna("").map(
            lambda r: {t for t in r.split(";") if t}.issubset(allowed))]
    df = df[df["stack_nb_occurrences"].fillna(0) >= args.min_occurrences]
    if df.empty:
        raise SystemExit("No negative Kyuma stacks matched the filters")

    font_map = {row.basename: row for row in load_font_rows(args.fonts_csv)}

    combined_manifest: list[dict[str, str]] = []
    contact_sheets: list[Path] = []
    for basename in kyuma:
        font = font_map.get(basename)
        if font is None:
            print(f"WARNING: font row not found for {basename}")
            continue
        sub = (
            df[df["basename"] == basename]
            .sort_values("stack_nb_occurrences", ascending=False)
            .head(args.top_per_font)
        )
        if sub.empty:
            continue
        font_manifest: list[dict[str, str]] = []
        for order, row in enumerate(sub.itertuples(index=False), start=1):
            stack = str(row.stack)
            stack_hash = hashlib.sha1(stack.encode("utf-8")).hexdigest()[:10]
            filename = f"{slugify(basename)}_{order:03d}_{stack_hash}.png"
            out_png = image_dir / filename
            ok, error = hb_view_render(
                font,
                stack,
                out_png,
                hb_view_bin=args.hb_view,
                margin=args.render_margin,
            )
            ewts = to_ewts(stack)
            label_text = ewts or str(row.codepoints)
            record = {
                "image": str(out_png.relative_to(args.out_dir)),
                "basename": basename,
                "stack": stack,
                "stack_ewts": ewts,
                "label_text": label_text,
                "codepoints": str(row.codepoints),
                "unicode_names": str(getattr(row, "unicode_names", "")),
                "stack_nb_occurrences": str(int(row.stack_nb_occurrences or 0)),
                "is_standard_tibetan": str(row.is_standard_tibetan),
                "support_class": str(row.support_class),
                "reason": str(row.reason),
                "subjoined_count": str(row.subjoined_count),
                "vowel_diacritic_count": str(row.vowel_diacritic_count),
                "placement_warnings": str(row.placement_warnings),
                "render_ok": str(ok),
                "render_error": error,
            }
            font_manifest.append(record)
            combined_manifest.append(record)
        sheet = make_contact_sheet(
            font_manifest,
            args.out_dir,
            title=f"{args.title_prefix}: {basename} (top {len(font_manifest)} by corpus frequency)",
            columns=args.columns,
            filename=f"contact_{slugify(basename)}.jpg",
        )
        if sheet:
            contact_sheets.append(sheet)
            print(f"Wrote {sheet}")

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined_manifest[0]))
        writer.writeheader()
        writer.writerows(combined_manifest)
    print(f"Wrote {manifest_path} ({len(combined_manifest)} rows, {len(contact_sheets)} sheets)")


if __name__ == "__main__":
    main()
