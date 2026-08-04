#!/usr/bin/env python3
"""Render one short Tibetan sample per font and split from a render plan."""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from synthetic_common import BENCHMARK_DIR, DEFAULT_RENDER_PLAN

DEFAULT_OUTPUT = Path("scripts/synthetic_benchmark/out/split_font_summary.pdf")
DEFAULT_SAMPLE_TEXT = "བོད་ཡིག་གི་ཚོད་ལྟ།"
SPLIT_ORDER = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render split/font summary PDF.")
    parser.add_argument("--render-plan", type=Path, default=DEFAULT_RENDER_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-text", default=DEFAULT_SAMPLE_TEXT)
    parser.add_argument("--lualatex", default="lualatex")
    return parser.parse_args()


def tex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def resolve_font_path(row: dict[str, object]) -> Path:
    rel_font_path = str(row.get("font_path") or "").strip()
    if rel_font_path:
        candidate = BENCHMARK_DIR / rel_font_path
        if candidate.exists():
            return candidate
    abs_font_path = Path(str(row.get("font_abs_path") or ""))
    if abs_font_path.exists():
        return abs_font_path
    return BENCHMARK_DIR / rel_font_path if rel_font_path else abs_font_path


def font_options(row: dict[str, object]) -> str:
    path = resolve_font_path(row)
    parts = [
        f"Path={{{path.parent.as_posix()}/}}",
        f"UprightFont={{{path.name}}}",
        "Renderer=HarfBuzz",
        "RawFeature={script=tibt}",
    ]
    face_index = str(row.get("ttc_face_index") or "").strip()
    if face_index:
        parts.append(f"FontIndex={face_index}")
    return ",\n    ".join(parts)


def load_fonts_by_split(render_plan: Path) -> dict[str, list[dict[str, object]]]:
    rows = pq.read_table(
        render_plan,
        columns=[
            "suggested_split",
            "font_name",
            "basename",
            "font_file",
            "font_path",
            "font_abs_path",
            "ttc_face_index",
            "script_8",
            "script",
        ],
    ).to_pylist()
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        split = str(row.get("suggested_split") or "unknown")
        name = str(row.get("font_name") or row.get("basename") or row.get("font_file") or "unknown")
        key = (split, name)
        by_key.setdefault(key, row)

    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (split, _name), row in by_key.items():
        by_split[split].append(row)
    for split_rows in by_split.values():
        split_rows.sort(key=lambda row: (str(row.get("script_8") or ""), str(row.get("font_name") or "")))
    return dict(by_split)


def make_tex(by_split: dict[str, list[dict[str, object]]], sample_text: str) -> str:
    parts = [
        r"\documentclass[10pt]{article}",
        r"\usepackage{fontspec}",
        r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{xcolor}",
        r"\pagestyle{plain}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{2pt}",
        r"\begin{document}",
        r"\begin{center}{\Large Synthetic Benchmark Fonts by Split}\end{center}",
    ]

    for split in SPLIT_ORDER:
        rows = by_split.get(split, [])
        if not rows:
            continue
        parts.append(rf"\section*{{{tex_escape(split)} ({len(rows)} fonts)}}")
        previous_script_8 = None
        for row in rows:
            script_8 = str(row.get("script_8") or "")
            if script_8 != previous_script_8:
                parts.append(rf"\subsection*{{{tex_escape(script_8)}}}")
                previous_script_8 = script_8
            label = " | ".join(
                item
                for item in [
                    str(row.get("font_name") or row.get("basename") or ""),
                    str(row.get("script") or ""),
                    str(row.get("font_file") or ""),
                ]
                if item
            )
            parts.append(r"\noindent")
            parts.append(rf"{{\scriptsize\color{{gray}}\texttt{{{tex_escape(label)}}}}}\\[-1pt]")
            parts.append(
                "{\\fontspec[\n    "
                + font_options(row)
                + "\n  ]{DUMMY}\\fontsize{18pt}{22pt}\\selectfont "
                + sample_text
                + r"}\par"
            )
        parts.append(r"\clearpage")
    parts.append(r"\end{document}")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.output.with_suffix(".tex")
    log_path = args.output.with_suffix(".log")
    by_split = load_fonts_by_split(args.render_plan)
    tex_path.write_text(make_tex(by_split, args.sample_text), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                args.lualatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-output-directory={args.output.parent}",
                str(tex_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path != args.output:
        pdf_path.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
