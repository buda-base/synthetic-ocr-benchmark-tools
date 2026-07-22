#!/usr/bin/env python3
"""Render shorthand *stacks* that auto-pass/fail coverage for manual review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyewts
from PIL import Image, ImageDraw

from coverage_common import (
    DEFAULT_FONTS_CSV,
    DEFAULT_OUT_DIR,
    DEFAULT_SHORTHAND_STACKS_CSV,
    hb_view_render,
    load_font_rows,
    slugify,
)


@lru_cache(maxsize=1)
def _ewts_converter() -> pyewts.pyewts:
    return pyewts.pyewts()


def to_ewts(text: str) -> str:
    try:
        return _ewts_converter().toWylie(text or "")
    except Exception as exc:  # pragma: no cover - defensive
        return f"<ewts_error:{type(exc).__name__}>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit shorthand stack renderings one stack at a time."
    )
    parser.add_argument("support_parquet", type=Path)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "shorthand_audit")
    parser.add_argument(
        "--kind",
        choices=("shorthand-pass", "shorthand-fail"),
        default="shorthand-pass",
    )
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hb-view", default="hb-view")
    parser.add_argument("--render-margin", type=int, default=80)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "placement_warning_count" not in df.columns:
        df["placement_warning_count"] = 0
        df["placement_warnings"] = ""
    if "placement_warnings" not in df.columns:
        df["placement_warnings"] = ""
    if "probe_source" not in df.columns:
        # shorthand-only parquet still has stack rows; treat all as shorthand.
        df["probe_source"] = "shorthand"
    df["final_ok"] = (df["ok"] == True) & (df["placement_warning_count"].fillna(0) == 0)  # noqa: E712
    return df


def filter_rows(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    shorthand = df["test_kind"] == "stack"
    # Prefer explicit probe_source tagging; if the parquet is shorthand-only,
    # every stack row is eligible.
    if (df["probe_source"] == "shorthand").any():
        shorthand = shorthand & (df["probe_source"] == "shorthand")
    if kind == "shorthand-pass":
        return df[shorthand & (df["final_ok"] == True)]  # noqa: E712
    return df[shorthand & (df["final_ok"] == False)]  # noqa: E712


def stack_complexity(row: pd.Series) -> int:
    text = str(row.get("stack") or "")
    score = int(row.get("complexity") or 0)
    score += text.count("\u0f39") * 3
    score += text.count("\u0f7e") * 2
    return score


def sample_rows(df: pd.DataFrame, sample_size: int, seed: int | None) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df
    random_state = seed if seed is not None else random.SystemRandom().randrange(2**31)
    ranked = df.assign(audit_rank=df.apply(stack_complexity, axis=1))
    selected = []
    groups = [
        group.sort_values("audit_rank", ascending=False)
        .head(max(sample_size, 1))
        .sample(frac=1, random_state=random_state + i)
        for i, (_, group) in enumerate(ranked.groupby("basename", sort=False))
    ]
    while groups and len(selected) < sample_size:
        next_groups = []
        for group in groups:
            if len(selected) >= sample_size:
                break
            selected.append(group.iloc[[0]])
            if len(group) > 1:
                next_groups.append(group.iloc[1:])
        groups = next_groups
    return pd.concat(selected).sample(frac=1, random_state=random_state).drop(columns=["audit_rank"])


def render_rows(
    rows: pd.DataFrame,
    font_map: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    image_dir = args.out_dir / "png"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        stack = str(row.stack)
        font = font_map.get(row.basename)
        stack_hash = hashlib.sha1(stack.encode("utf-8")).hexdigest()[:10]
        filename = f"{i:04d}_{slugify(row.basename)}_{stack_hash}.png"
        out_png = image_dir / filename
        ok, error = (False, "font row not found")
        if font is not None:
            ok, error = hb_view_render(
                font,
                stack,
                out_png,
                hb_view_bin=args.hb_view,
                margin=args.render_margin,
            )
        final_ok = bool(row.final_ok)
        manifest.append(
            {
                "image": str(out_png.relative_to(args.out_dir)),
                "render_ok": str(ok),
                "render_error": error,
                "basename": row.basename,
                "skt_ok": str(getattr(row, "skt_ok", "")),
                "stack": stack,
                "stack_ewts": to_ewts(stack),
                "codepoints": getattr(row, "codepoints", ""),
                "auto_ok": str(row.ok),
                "final_ok": str(final_ok),
                "reason": getattr(row, "reason", ""),
                "complexity": str(getattr(row, "complexity", "")),
                "subjoined_count": str(getattr(row, "subjoined_count", "")),
                "vowel_diacritic_count": str(getattr(row, "vowel_diacritic_count", "")),
                "placement_warnings": getattr(row, "placement_warnings", "") or "",
                "kind": args.kind,
            }
        )
    return manifest


def make_contact_sheet(manifest: list[dict[str, str]], out_dir: Path, *, columns: int) -> Path | None:
    if not manifest:
        return None
    cell_w, cell_h = 460, 400
    image_h = 260
    text_y = image_h + 16
    rows = (len(manifest) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(manifest):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        png_path = out_dir / item["image"]
        if png_path.exists():
            with Image.open(png_path) as img:
                img = img.convert("RGB")
                img.thumbnail((cell_w - 20, image_h))
                sheet.paste(img, (x + 10, y + 10))
        final_ok = item["final_ok"] == "True"
        label_fill = "black" if final_ok else "red"
        if not final_ok:
            draw.rectangle((x + 2, y + 2, x + cell_w - 3, y + cell_h - 3), outline="red", width=3)
        # Label with shorthand-stack EWTS only (not the expanded long form).
        label = (
            f"{i + 1}. {item['basename']} skt={item['skt_ok']} final={item['final_ok']}\n"
            f"{item['stack_ewts']}\n"
            f"sub={item['subjoined_count']} vowel={item['vowel_diacritic_count']} "
            f"render={item['render_ok']}"
        )
        draw.multiline_text((x + 10, y + text_y), label, fill=label_fill, spacing=3)
    out_path = out_dir / "contact_sheet.jpg"
    sheet.save(out_path, quality=90)
    return out_path


def write_manifest(manifest: list[dict[str, str]], path: Path) -> None:
    if not manifest:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    args = parse_args()
    if not args.support_parquet.is_file():
        raise SystemExit(
            f"Missing support parquet: {args.support_parquet}\n"
            "Build it first, e.g.:\n"
            "  python coverage_report/export_shorthand_stacks.py\n"
            "  python coverage_report/build_support_dataset.py \\\n"
            "    --shorthand-only \\\n"
            "    --output coverage_report/out/shorthand_support.parquet\n"
            "Then re-run this audit on that parquet."
        )
    if not DEFAULT_SHORTHAND_STACKS_CSV.is_file():
        print(
            f"NOTE: {DEFAULT_SHORTHAND_STACKS_CSV} missing; "
            "run export_shorthand_stacks.py if probes are stale."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(pd.read_parquet(args.support_parquet))
    df = filter_rows(df, args.kind)
    if df.empty:
        raise SystemExit(f"No rows matched audit kind: {args.kind}")
    df = sample_rows(df, args.sample_size, args.seed)
    font_map = {row.basename: row for row in load_font_rows(args.fonts_csv)}
    manifest = render_rows(df, font_map, args)
    manifest_path = args.out_dir / "manifest.csv"
    write_manifest(manifest, manifest_path)
    contact_sheet = make_contact_sheet(manifest, args.out_dir, columns=args.columns)
    print(f"Wrote {manifest_path} ({len(manifest)} stack row(s))")
    if contact_sheet:
        print(f"Wrote {contact_sheet}")


if __name__ == "__main__":
    main()
