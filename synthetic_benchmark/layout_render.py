#!/usr/bin/env python3
"""v6 layout-recipe renderer core: per-page geometry + exact per-line labels.

Renders one text block (a single page, or one slab of a composite) with the
geometry sampled by :mod:`layout_recipe`, filling the frame to ~N lines, and
reconstructs the per-visual-line transcription using the SAME LuaLaTeX
syllable-marker machinery as the production ``render_batches`` pipeline (so the
label convention is identical). Composition of blocks into composites is done at
the image level by :mod:`layout_compose`.

This is deliberately standalone so it can be validated against local lualatex
before being wired into ``build_render_plan`` / ``render_batches``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import render_batches as rb  # reused: add_marked_breakpoints, split_marker_units, parse_*
from synthetic_common import tex_escape
import layout_recipe as lr
from layout_compose import crop_to_ink, pad_to_margins, stack_blocks, photo_composite

# Macro preamble replicated from render_batches.make_tex so the LuaLaTeX page /
# line maps emit the same syllable-marker ids we reconstruct labels from.
_SYLL_MACROS = r"""
\newcount\syllid
\syllid=0
\newattribute\syllattr
\def\syllmark{%
  \global\advance\syllid by 1
  \directlua{
    local n = node.new("kern")
    n.kern = 0
    node.set_attribute(n, tex.attribute.syllattr, tex.count.syllid)
    local nest = tex.nest[tex.nest.ptr]
    if nest and nest.tail then
      node.insert_after(nest.head, nest.tail, n)
      nest.tail = n
    else
      node.write(n)
    end
  }%
}
\makeatletter
\newattribute\ycattr
\newdimen\ycheadraise
\def\ycsize{10pt}
\def\yccalcheadraise{\global\ycheadraise=0pt}
\def\yigchung#1{#1}
\makeatother
"""


@dataclass
class BlockRender:
    image: np.ndarray          # grayscale, 255 = paper, 0 = ink, cropped to ink
    label_lines: list[str]     # one entry per rendered visual line (blanks skipped)
    realized_lines: int
    pitch_px: float


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _build_block_tex(
    *,
    font_path: Path,
    font_pt: float,
    baseline_pt: float,
    paper_w_mm: float,
    paper_h_mm: float,
    pad_mm: float,
    tw_mm: float,
    th_mm: float,
    block_id: str,
    body: str,
    tight: bool,
    page_map_path: Path,
) -> str:
    lua = rb.lua_page_map_code(page_map_path, page_map_path.with_suffix(".lines.csv"))
    lineskiplimit = "-20pt" if tight else "0.5pt"
    return rf"""\documentclass{{article}}
\usepackage{{fontspec}}
\usepackage[paperwidth={_fmt(paper_w_mm)}mm,paperheight={_fmt(paper_h_mm)}mm,left={_fmt(pad_mm)}mm,top={_fmt(pad_mm)}mm,textwidth={_fmt(tw_mm)}mm,textheight={_fmt(th_mm)}mm]{{geometry}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0pt}}
\pagestyle{{empty}}
\lineskiplimit={lineskiplimit}
\emergencystretch=1em
\tolerance=9999
\pretolerance=9999
{_SYLL_MACROS}
{lua}
\setmainfont{{{font_path.name}}}[Path={font_path.parent.as_posix()}/,Renderer=HarfBuzz,RawFeature={{script=tibt}}]
\begin{{document}}
\fontsize{{{_fmt(font_pt)}pt}}{{{_fmt(baseline_pt)}pt}}\selectfont
\directlua{{synthetic_set_chunk("{block_id}")}}
\noindent {body}
\clearpage
\end{{document}}
"""


def _rebuild_lines(plain: str, line_marker_ids: list[list[int]]) -> list[str]:
    """Reconstruct per-visual-line text from syllable-marker ids (blanks skipped)."""
    units = rb.split_marker_units(plain)
    lines: list[str] = []
    previous_end = 0
    for ids in line_marker_ids:
        if not ids:
            continue
        end = min(max(ids), len(units))
        if end <= previous_end:
            continue
        line = "".join(units[previous_end:end]).strip()
        if line:
            lines.append(line)
        previous_end = end
    return lines


def render_block(
    block: lr.BlockSpec,
    text: str,
    *,
    box_w_px: float,
    font_path: Path,
    dpi: int,
    workdir: Path,
    block_id: str,
    pad_mm: float = 6.0,
) -> BlockRender | None:
    """Render a single text block.

    ``box_w_px`` sets the text column width; ``text`` is over-provided and the
    frame height (``n_lines`` pitches) caps page 1 to ~N visual lines. The
    realized aspect emerges from wrapping, matching the "estimate" fill mode.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    pt_per_px = 72.0 / dpi
    mm_per_px = 25.4 / dpi
    font_pt = block.font_px * pt_per_px
    baseline_pt = block.line_pitch_px * pt_per_px
    th_px = max(block.line_pitch_px, block.n_lines * block.line_pitch_px)
    return _render_with_box(
        block, text, font_path=font_path, dpi=dpi, workdir=workdir,
        block_id=block_id, pad_mm=pad_mm, tw_px=box_w_px, th_px=th_px,
        font_pt=font_pt, baseline_pt=baseline_pt, mm_per_px=mm_per_px,
    )


def _render_with_box(
    block, text, *, font_path, dpi, workdir, block_id, pad_mm,
    tw_px, th_px, font_pt, baseline_pt, mm_per_px,
) -> BlockRender | None:
    tw_mm = tw_px * mm_per_px
    th_mm = th_px * mm_per_px
    paper_w = tw_mm + 2 * pad_mm
    paper_h = th_mm + 2 * pad_mm
    body = rb.add_marked_breakpoints(tex_escape(text))
    tight = block.gap_frac <= 0.15
    tex = _build_block_tex(
        font_path=font_path, font_pt=font_pt, baseline_pt=baseline_pt,
        paper_w_mm=paper_w, paper_h_mm=paper_h, pad_mm=pad_mm,
        tw_mm=tw_mm, th_mm=th_mm, block_id=block_id, body=body,
        tight=tight, page_map_path=workdir / f"{block_id}.pages.csv",
    )
    texf = workdir / f"{block_id}.tex"
    texf.write_text(tex, encoding="utf-8")
    pdf = workdir / f"{block_id}.pdf"
    raw = workdir / f"{block_id}.jpg"
    for p in (pdf, raw, workdir / f"{block_id}.pages.csv", workdir / f"{block_id}.pages.lines.csv"):
        p.unlink(missing_ok=True)
    r = subprocess.run(
        ["lualatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={workdir}", str(texf)],
        capture_output=True, text=True,
    )
    if not pdf.exists():
        print(f"  [render_block] lualatex failed for {block_id}:\n{r.stdout[-800:]}")
        return None
    subprocess.run(
        ["pdftoppm", "-f", "1", "-l", "1", "-singlefile", "-r", str(dpi),
         "-jpeg", "-gray", str(pdf), str(workdir / block_id)],
        capture_output=True,
    )
    img = cv2.imread(str(raw), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    page_map = rb.parse_page_map(workdir / f"{block_id}.pages.csv")
    line_map = rb.parse_line_map(workdir / f"{block_id}.pages.lines.csv")
    pages = page_map.get(block_id, [])
    first_page_no = pages[0]["physical_page"] if pages else 1
    marker_ids = line_map.get((block_id, first_page_no), [])
    labels = _rebuild_lines(rb.strip_yc_tags(text) if hasattr(rb, "strip_yc_tags") else text, marker_ids)
    cropped = crop_to_ink(img)
    return BlockRender(
        image=cropped,
        label_lines=labels,
        realized_lines=len(labels),
        pitch_px=block.line_pitch_px,
    )


# ---------------------------------------------------------------------------
# Ragged / blank body construction.
# ---------------------------------------------------------------------------


def build_body(text: str, block: lr.BlockSpec, rng) -> str:
    r"""Return LaTeX body text with optional forced short (ragged) / blank lines.

    Non-ragged pages return the plain text (auto-wrapped to fill the column).
    Ragged pages intersperse forced 1-3 syllable short lines (``\par``) among
    full auto-wrapped runs; blank pages add occasional empty rendered lines
    (glyph-less, so the label machinery skips them).
    """
    if not (block.ragged or block.blank_lines):
        return rb.add_marked_breakpoints(tex_escape(text))

    units = rb.split_marker_units(text)
    if not units:
        return rb.add_marked_breakpoints(tex_escape(text))

    out: list[str] = []
    i = 0
    n = len(units)
    while i < n:
        if block.ragged and rng.random() < block.ragged_fraction:
            k = rng.randint(1, 3)
            chunk = "".join(units[i : i + k])
            out.append(rb.add_marked_breakpoints(tex_escape(chunk)) + r"\par")
            i += k
        else:
            # A full-width run: 8-20 units auto-wrapped.
            k = rng.randint(8, 20)
            chunk = "".join(units[i : i + k])
            out.append(rb.add_marked_breakpoints(tex_escape(chunk)) + r"\par")
            i += k
        if block.blank_lines and rng.random() < 0.12:
            out.append(r"\vspace{\baselineskip}\mbox{}\par")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Page assembly (single / stack / photo).
# ---------------------------------------------------------------------------


def _content_width_px(layout: lr.PageLayout, eff_lines: int, block: lr.BlockSpec) -> float:
    width, height, _ = lr.capped_dimensions_px(layout)
    shorter = min(width, height)
    m = block.margins
    return max(60.0, width - (m.left + m.right) * shorter)


def render_page(
    layout: lr.PageLayout,
    texts: list[str],
    *,
    font_path: Path,
    dpi: int,
    workdir: Path,
    seed: int,
):
    """Render a full page (single/stack/photo). ``texts`` supplies one
    over-provided text per block. Returns (gray_image, label, info)."""
    from dataclasses import replace

    eff = lr.effective_line_counts(layout)
    rng = __import__("random").Random(f"{seed}:{layout.page_id}")
    block_imgs: list[np.ndarray] = []
    block_labels: list[list[str]] = []
    pitches: list[float] = []

    for bi, block in enumerate(layout.blocks):
        b = replace(block, n_lines=eff[bi])
        if layout.kind == "single":
            box_w = _content_width_px(layout, eff[bi], b)
        else:
            m = b.margins
            box_w = max(60.0, float(layout.target_width_px or 1200.0) * (1.0 - m.left - m.right))
        body = build_body(texts[bi], b, rng)
        br = _render_prebuilt_body(
            b, body, texts[bi], box_w_px=box_w, font_path=font_path,
            dpi=dpi, workdir=workdir, block_id=f"p{layout.page_id}_b{bi}",
        )
        if br is None:
            return None
        # Apply this block's margins (single page: page margins; composite: per-block).
        padded, _ = pad_to_margins(br.image, _margins_tuple(b.margins))
        block_imgs.append(padded)
        block_labels.append(br.label_lines)
        pitches.append(br.pitch_px)

    if layout.kind == "single":
        image = block_imgs[0]
    elif layout.kind == "stack":
        image, _ = stack_blocks(block_imgs, layout.separators, pitches)
    else:  # photo
        image, _ = photo_composite(block_imgs, seed=seed ^ layout.page_id)

    label = "\n".join("\n".join(lbls) for lbls in block_labels if lbls)
    info = {
        "kind": layout.kind,
        "aspect_sampled": round(layout.aspect, 4),
        "aspect_realized": round(image.shape[1] / max(1, image.shape[0]), 4),
        "blocks": len(layout.blocks),
        "realized_lines": [len(l) for l in block_labels],
        "font_px": [round(b.font_px, 1) for b in layout.blocks],
        "image_wh": [int(image.shape[1]), int(image.shape[0])],
    }
    return image, label, info


def _margins_tuple(m: lr.Margins):
    return (m.left, m.right, m.top, m.bottom)


def _render_prebuilt_body(
    block, body, plain_text, *, box_w_px, font_path, dpi, workdir, block_id, pad_mm: float = 6.0,
) -> BlockRender | None:
    """Like render_block but the LaTeX body is already built (ragged/blank aware)."""
    workdir.mkdir(parents=True, exist_ok=True)
    pt_per_px = 72.0 / dpi
    mm_per_px = 25.4 / dpi
    font_pt = block.font_px * pt_per_px
    baseline_pt = block.line_pitch_px * pt_per_px
    th_px = max(block.line_pitch_px, block.n_lines * block.line_pitch_px)
    tw_mm = box_w_px * mm_per_px
    th_mm = th_px * mm_per_px
    paper_w = tw_mm + 2 * pad_mm
    paper_h = th_mm + 2 * pad_mm
    tight = block.gap_frac <= 0.15
    tex = _build_block_tex(
        font_path=font_path, font_pt=font_pt, baseline_pt=baseline_pt,
        paper_w_mm=paper_w, paper_h_mm=paper_h, pad_mm=pad_mm,
        tw_mm=tw_mm, th_mm=th_mm, block_id=block_id, body=body,
        tight=tight, page_map_path=workdir / f"{block_id}.pages.csv",
    )
    texf = workdir / f"{block_id}.tex"
    texf.write_text(tex, encoding="utf-8")
    pdf = workdir / f"{block_id}.pdf"
    raw = workdir / f"{block_id}.jpg"
    for p in (pdf, raw, workdir / f"{block_id}.pages.csv", workdir / f"{block_id}.pages.lines.csv"):
        p.unlink(missing_ok=True)
    r = subprocess.run(
        ["lualatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={workdir}", str(texf)],
        capture_output=True, text=True,
    )
    if not pdf.exists():
        print(f"  [render] lualatex failed for {block_id}:\n{r.stdout[-800:]}")
        return None
    subprocess.run(
        ["pdftoppm", "-f", "1", "-l", "1", "-singlefile", "-r", str(dpi),
         "-jpeg", "-gray", str(pdf), str(workdir / block_id)],
        capture_output=True,
    )
    img = cv2.imread(str(raw), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    line_map = rb.parse_line_map(workdir / f"{block_id}.pages.lines.csv")
    page_map = rb.parse_page_map(workdir / f"{block_id}.pages.csv")
    pages = page_map.get(block_id, [])
    first_page_no = pages[0]["physical_page"] if pages else 1
    marker_ids = line_map.get((block_id, first_page_no), [])
    labels = _rebuild_lines(plain_text, marker_ids)
    return BlockRender(
        image=crop_to_ink(img),
        label_lines=labels,
        realized_lines=len(labels),
        pitch_px=block.line_pitch_px,
    )
