#!/usr/bin/env python3
"""Render a stratified sample of stacks that flipped negative -> positive.

A "flip" is a (font, stack) that is ``ok`` in the relaxed support parquet only
because a style-gated geometry rule (``mid_stem`` and/or
``top_mark_horizontal_misalignment``) was skipped for that font's style. We
recompute the *unrelaxed* placement warnings from the stored glyph boxes (no
reshaping), so a flip is any ok row whose unrelaxed warnings intersect the
font's relaxed-rule set. Rendering these lets us eyeball for false positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from coverage_common import (
    DEFAULT_FONTS_CSV,
    DEFAULT_OUT_DIR,
    apply_geometry_relaxation,
    detect_placement_warnings,
    hb_view_render,
    load_font_rows,
    load_relaxation_map_by_basename,
    slugify,
)

try:
    import pyewts

    @lru_cache(maxsize=1)
    def _ewts() -> "pyewts.pyewts":
        return pyewts.pyewts()

    def to_ewts(text: str) -> str:
        try:
            return _ewts().toWylie(text or "")
        except Exception:
            return ""
except Exception:
    def to_ewts(text: str) -> str:  # type: ignore[misc]
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("support_parquet", type=Path)
    p.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "flip_audit")
    p.add_argument("--per-rule", type=int, default=90, help="Cells per flip rule.")
    p.add_argument("--per-font-cap", type=int, default=6, help="Max cells per font per rule.")
    p.add_argument("--min-occurrences", type=int, default=1)
    p.add_argument("--standard-only", action="store_true")
    p.add_argument("--columns", type=int, default=6)
    p.add_argument("--render-margin", type=int, default=60)
    p.add_argument("--hb-view", default="hb-view")
    return p.parse_args()


def unrelaxed_warnings(stack: str, boxes_json: str) -> set[str]:
    if not boxes_json:
        return set()
    try:
        boxes = json.loads(boxes_json)
    except Exception:
        return set()
    return set(detect_placement_warnings(stack, boxes, frozenset()))


def make_sheet(cells, out_dir, *, title, columns, filename):
    if not cells:
        return None
    cell_w, cell_h = 430, 380
    image_h = 220
    header_h = 40
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black")
    for i, item in enumerate(cells):
        x = (i % columns) * cell_w
        y = header_h + (i // columns) * cell_h
        png = out_dir / item["image"]
        if png.exists():
            with Image.open(png) as img:
                img = img.convert("RGB")
                img.thumbnail((cell_w - 20, image_h))
                sheet.paste(img, (x + 10, y + 10))
        draw.rectangle((x + 2, y + 2, x + cell_w - 3, y + cell_h - 3), outline="red", width=2)
        label = (
            f"{i + 1}. {item['label_text']}  occ={item['occ']}\n"
            f"{item['basename'][:34]}\n"
            f"flip={item['flip_rule']}"
        )
        draw.multiline_text((x + 10, y + image_h + 26), label, fill="black", spacing=4)
    out = out_dir / filename
    sheet.save(out, quality=90)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "png"
    image_dir.mkdir(parents=True, exist_ok=True)

    relax = load_relaxation_map_by_basename()
    relaxed_fonts = {b for b, r in relax.items() if r}

    cols = ["test_kind", "basename", "stack", "ok", "is_standard_tibetan",
            "stack_nb_occurrences", "codepoints", "glyph_boxes"]
    df = pd.read_parquet(args.support_parquet, columns=cols)
    df = df[(df["test_kind"] == "stack") & (df["ok"]) & (df["basename"].isin(relaxed_fonts))].copy()
    if args.standard_only:
        df = df[df["is_standard_tibetan"] == 1]
    occ_map = (
        pd.read_parquet(args.support_parquet, columns=["stack", "stack_nb_occurrences"])
        .groupby("stack")["stack_nb_occurrences"].max()
    )
    df["occ"] = df["stack"].map(occ_map).fillna(0).astype(int)
    df = df[df["occ"] >= args.min_occurrences]
    df = df.sort_values("occ", ascending=False)

    rules = ["mid_stem", "top_mark_horizontal_misalignment"]
    collected: dict[str, list] = {r: [] for r in rules}
    per_font_count: dict[tuple[str, str], int] = {}

    for row in df.itertuples(index=False):
        if all(len(collected[r]) >= args.per_rule for r in rules):
            break
        font_relaxed = relax.get(row.basename, frozenset())
        warns = unrelaxed_warnings(row.stack, row.glyph_boxes)
        flip_rules = warns & font_relaxed
        if not flip_rules:
            continue
        for r in rules:
            if r not in flip_rules or len(collected[r]) >= args.per_rule:
                continue
            key = (r, row.basename)
            if per_font_count.get(key, 0) >= args.per_font_cap:
                continue
            per_font_count[key] = per_font_count.get(key, 0) + 1
            collected[r].append({
                "basename": row.basename,
                "stack": row.stack,
                "occ": int(row.occ),
                "codepoints": str(row.codepoints),
                "flip_rule": ";".join(sorted(flip_rules)),
            })

    font_map = {f.basename: f for f in apply_geometry_relaxation(load_font_rows(args.fonts_csv))}
    manifest = []
    sheets = []
    for r in rules:
        cells = []
        for order, item in enumerate(collected[r], 1):
            font = font_map.get(item["basename"])
            if font is None:
                continue
            h = hashlib.sha1((item["basename"] + item["stack"]).encode()).hexdigest()[:10]
            fname = f"{slugify(item['basename'])}_{order:03d}_{h}.png"
            out_png = image_dir / fname
            ok, err = hb_view_render(font, item["stack"], out_png,
                                     hb_view_bin=args.hb_view, margin=args.render_margin)
            ewts = to_ewts(item["stack"])
            rec = dict(item)
            rec["image"] = str(out_png.relative_to(args.out_dir))
            rec["label_text"] = ewts or item["codepoints"]
            rec["render_ok"] = str(ok)
            rec["render_error"] = err
            cells.append(rec)
            manifest.append(rec)
        sheet = make_sheet(cells, args.out_dir, title=f"negative->positive flips on {r} (n={len(cells)})",
                           columns=args.columns, filename=f"flips_{r}.jpg")
        if sheet:
            sheets.append(sheet)
            print(f"Wrote {sheet}  ({len(cells)} cells, "
                  f"{len({c['basename'] for c in cells})} fonts)")

    man = args.out_dir / "manifest.csv"
    pd.DataFrame(manifest).to_csv(man, index=False)
    print(f"Wrote {man} ({len(manifest)} rows, {len(sheets)} sheets)")


if __name__ == "__main__":
    main()
