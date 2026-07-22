#!/usr/bin/env python3
"""Build the unified shorthand lexicon CSV from vendored rKTs and TibSchol sources."""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from synthetic_common import normalize_bocorpus_text

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "shorthands"
DEFAULT_RKTS = DEFAULT_DATA_DIR / "rkts_abb.xml"
DEFAULT_TIBSCHOL = DEFAULT_DATA_DIR / "tibschol_abbr.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "shorthands.csv"

TSHEG = "་"
DELIMITED_TSHEG = "༌"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified Tibetan shorthand lexicon CSV.")
    parser.add_argument("--rkts-xml", type=Path, default=DEFAULT_RKTS)
    parser.add_argument("--tibschol-csv", type=Path, default=DEFAULT_TIBSCHOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_form(text: str) -> str:
    text = (text or "").strip().replace("_", "")
    text = text.replace(DELIMITED_TSHEG, TSHEG)
    text = normalize_bocorpus_text(text)
    return text.strip(TSHEG + " \t")


def iter_rkts_pairs(path: Path) -> list[tuple[str, str, str]]:
    root = ET.parse(path).getroot()
    pairs: list[tuple[str, str, str]] = []
    for item in root.findall("item"):
        tib = item.findtext("tib") or ""
        abb = item.findtext("abb") or ""
        long_form = normalize_form(tib)
        shorthand = normalize_form(abb)
        if not long_form or not shorthand or long_form == shorthand:
            continue
        pairs.append((long_form, shorthand, "rkts"))
    return pairs


def iter_tibschol_pairs(path: Path) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Columns are named with Tibetan labels in the header.
            long_raw = ""
            short_raw = ""
            for key, value in row.items():
                key_l = (key or "").lower()
                if "expan" in key_l and "unicode" in key_l:
                    long_raw = value or ""
                elif key_l.startswith("abb") and "unicode" in key_l:
                    short_raw = value or ""
            if not long_raw or not short_raw:
                # Fallback positional: Abb.Wylie, Abb.Unicode, Expan.Wylie, Expan.Unicode
                values = list(row.values())
                if len(values) >= 4:
                    short_raw = values[1] or ""
                    long_raw = values[3] or ""
            long_form = normalize_form(long_raw)
            shorthand = normalize_form(short_raw)
            if not long_form or not shorthand or long_form == shorthand:
                continue
            # Skip digit-leading novelty forms for OCR manuscript realism.
            if re.search(r"[0-9༠-༩]", shorthand):
                continue
            pairs.append((long_form, shorthand, "tibschol"))
    return pairs


def main() -> None:
    args = parse_args()
    if not args.rkts_xml.is_file():
        raise SystemExit(f"Missing rKTs XML: {args.rkts_xml}")
    if not args.tibschol_csv.is_file():
        raise SystemExit(f"Missing TibSchol CSV: {args.tibschol_csv}")

    pairs = iter_rkts_pairs(args.rkts_xml) + iter_tibschol_pairs(args.tibschol_csv)
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for long_form, shorthand, source in pairs:
        key = (long_form, shorthand)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"long_form": long_form, "shorthand": shorthand, "source": source})

    rows.sort(key=lambda row: (-len(row["long_form"]), row["long_form"], row["shorthand"], row["source"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["long_form", "shorthand", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} pair(s))")


if __name__ == "__main__":
    main()
