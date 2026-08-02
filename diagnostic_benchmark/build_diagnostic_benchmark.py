#!/usr/bin/env python3
"""Build a small controlled diagnostic Tibetan OCR benchmark (OFAT design).

This is *not* the coverage-maximizing 500k dataset. It is a scientific
instrument: a few hundred cleanly rendered page images with exact, noise-free
ground truth, designed so that OCR character-error-rate (CER) can be studied as
a response to *one difficulty axis at a time* (One-Factor-At-A-Time), holding
everything else at a fixed "center" configuration.

Hard constraints implemented here (see the generated README for the rationale):

* A single Uchen font for the whole benchmark (``--font``), one clean absolute
  font size, one page width, one image pixel width, one line-spacing factor.
* No augraphy ink/paper effects, no font-level augmentation, no shorthands, no
  rotation, no TPS. The only image processing applied is the *minimum* raster
  naturalization the production renderer always applies (profile A: a slight
  sub-pixel warp, a faint Gaussian blur, fine luminance noise, and sparse white
  ink dropout). Augmentation is strictly off by default.
* Ground truth is the LuaLaTeX-rendered text reconstructed line-for-line from
  TeX syllable markers, then normalized with the *exact* Unicode + graphical
  normalization used by ``bec-ocr-training`` (``normalize_unicode_text``), so
  the parquet drops straight into that eval harness.

The output parquet has exactly the columns consumed by
``bec-ocr-training/scripts/paddle_reeval_full.py`` plus the OFAT metadata:

    file_name, transcription, image, script, technology, benchmark_group,
    axis, axis_bin, axis_value

A ``manifest.csv`` records per-image provenance (font, chunk id, seed, realized
metrics), and a ``provenance.json`` records the full configuration so the whole
thing regenerates from one seeded command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SYNTHETIC_DIR = REPO_ROOT / "synthetic_benchmark"
BENCHMARK_DIR = REPO_ROOT / "benchmark_gen"
COVERAGE_DIR = REPO_ROOT / "coverage_report"
DEFAULT_FONTS_CSV = BENCHMARK_DIR / "digital_fonts.filtered.csv"
DEFAULT_CHUNKS_PARQUET = SYNTHETIC_DIR / "out" / "bocorpus_chunks.parquet"
DEFAULT_STACKS_CSV = COVERAGE_DIR / "bocorpus_stacks.csv"
DEFAULT_SUPPORT_PARQUET = COVERAGE_DIR / "out" / "stack_support.parquet"
DEFAULT_OUT_DIR = SCRIPT_DIR / "out"

# Reuse the main pipeline's coverage gate (HarfBuzz + placement rules).
if str(SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC_DIR))
from synthetic_common import (  # noqa: E402
    load_supported_stacks,
    tokenize_tibetan_stacks,
)

TSHEG = "\u0f0b"
SHAD = "\u0f0d"

# ---------------------------------------------------------------------------
# Vendored canonical Tibetan Unicode normalization.
#
# Copied verbatim (behaviour-identical) from
# ``bec-ocr-training/bdrc_ocr/tibetan_normalize.py`` so ground-truth strings are
# in the *exact* normal form the training/eval harness scores against
# (``normalize_unicode_text``). Keeping a local copy means this builder has no
# dependency on botok and produces GT that is byte-identical to what the scorer
# expects.
# ---------------------------------------------------------------------------

from enum import Enum


class _Cats(Enum):
    Other = 0
    Base = 1
    Subscript = 2
    BottomVowel = 3
    BottomMark = 4
    TopVowel = 5
    TopMark = 6
    RightMark = 7


_CATEGORIES = (
    [_Cats.Other]
    + [_Cats.Base]
    + [_Cats.Other] * 22
    + [_Cats.BottomVowel] * 2
    + [_Cats.Other] * 6
    + [_Cats.Base] * 20
    + [_Cats.Other]
    + [_Cats.BottomMark]
    + [_Cats.Other]
    + [_Cats.BottomMark]
    + [_Cats.Other]
    + [_Cats.Subscript]
    + [_Cats.Other] * 4
    + [_Cats.RightMark]
    + [_Cats.Other]
    + [_Cats.Base] * 45
    + [_Cats.Other] * 4
    + [_Cats.BottomVowel]
    + [_Cats.TopVowel]
    + [_Cats.TopVowel]
    + [_Cats.BottomVowel] * 2
    + [_Cats.TopVowel] * 8
    + [_Cats.TopMark]
    + [_Cats.RightMark]
    + [_Cats.TopVowel] * 2
    + [_Cats.TopMark] * 2
    + [_Cats.BottomMark]
    + [_Cats.Other]
    + [_Cats.TopMark] * 2
    + [_Cats.Base] * 2
    + [_Cats.Base]
    + [_Cats.Other]
    + [_Cats.Base]
    + [_Cats.Subscript] * 48
)


def _charcat(c: str) -> _Cats:
    o = ord(c)
    if 0x0F00 <= o <= 0x0FBC:
        return _CATEGORIES[o - 0x0F00]
    return _Cats.Other


def _unicode_reorder(txt: str):
    charcats = [_charcat(c) for c in txt]
    i = 0
    res = []
    valid = True
    while i < len(charcats):
        c = charcats[i]
        if c != _Cats.Base:
            if c.value > _Cats.Base.value:
                valid = False
            res.append(txt[i])
            i += 1
            continue
        j = i + 1
        while j < len(charcats) and charcats[j].value > _Cats.Base.value:
            j += 1
        newindices = sorted(range(i, j), key=lambda e: (charcats[e].value, e))
        res.append("".join(txt[n] for n in newindices))
        i = j
    return "".join(res), valid


def _is_vowel(char: str) -> bool:
    return bool(re.search(r"[\u0f71-\u0f84]", char))


def _is_suffix(char: str) -> bool:
    return bool(re.search(r"[\u0f90-\u0fbc]", char))


def _normalize_invalid_start_string(s: str) -> str:
    if len(s) < 2:
        return s
    if _is_vowel(s[0]) and not _is_vowel(s[1]) and not _is_suffix(s[1]):
        return s[1] + s[0] + (s[2:] if len(s) > 2 else "")
    if _is_suffix(s[0]):
        return s[1:]
    return s


def _normalize_unicode(s: str, form: str = "nfd") -> str:
    s = s.replace("\u0f73", "\u0f71\u0f72")
    s = s.replace("\u0f75", "\u0f71\u0f74")
    s = s.replace("\u0f77", "\u0fb2\u0f71\u0f80")
    s = s.replace("\u0f79", "\u0fb3\u0f71\u0f80")
    s = s.replace("\u0f81", "\u0f71\u0f80")
    if form == "nfd":
        s = s.replace("\u0f43", "\u0f42\u0fb7")
        s = s.replace("\u0f48", "\u0f47\u0fb7")
        s = s.replace("\u0f4d", "\u0f4c\u0fb7")
        s = s.replace("\u0f52", "\u0f51\u0fb7")
        s = s.replace("\u0f57", "\u0f56\u0fb7")
        s = s.replace("\u0f5c", "\u0f5b\u0fb7")
        s = s.replace("\u0f69", "\u0f40\u0fb5")
        s = s.replace("\u0f76", "\u0fb2\u0f80")
        s = s.replace("\u0f78", "\u0fb3\u0f80")
        s = s.replace("\u0f93", "\u0f92\u0fb7")
        s = s.replace("\u0f98", "\u0f97\u0fb7")
        s = s.replace("\u0f9d", "\u0f9c\u0fb7")
        s = s.replace("\u0fa2", "\u0fa1\u0fb7")
        s = s.replace("\u0fa7", "\u0fa6\u0fb7")
        s = s.replace("\u0fac", "\u0fab\u0fb7")
        s = s.replace("\u0fb9", "\u0f90\u0fb5")
    s = s.replace("\u0f00", "\u0f68\u0f7c\u0f7e")
    s = s.replace("ཅ༹", "ཙ")
    s = s.replace("ཆ༹", "ཚ")
    s = s.replace("ཇ༹", "ཛ")
    s, _valid = _unicode_reorder(s)
    s = re.sub("\u0f6a(?![\u0f90-\u0f97\u0f9a-\u0fac\u0fae\u0faf\u0fb4-\u0fbc])", "ར", s)
    s = _normalize_invalid_start_string(s)
    return s


def _normalize_graphical(s: str) -> str:
    s = s.replace("\u0f0c", "\u0f0b")
    s = s.replace("\u0f0e", "\u0f0d\u0f0d")
    s = s.replace("\u0f38", "\u0f27")
    s = s.replace("\u0f7a\u0f7a", "\u0f7b")
    s = s.replace("\u0f7c\u0f7c", "\u0f7d")
    s = s.replace("༇", "࿓།࿒།")
    s = re.sub(r"[\u0f71]([\u0f8d-\u0fac\u0fae\u0fb0\u0fb3-\u0fbc])", "\u0fb0\\1", s)
    s = re.sub(r"[\u0fb0]([^\u0f8d-\u0fac\u0fae\u0fb0\u0fb3-\u0fbc]|$)", "\u0f71\\1", s)
    return s


def normalize_unicode_text(text: str | None) -> str:
    """Canonical Tibetan Unicode form (NFD + reorder, then graphical folding)."""
    if not text:
        return ""
    return _normalize_graphical(_normalize_unicode(text))


# ---------------------------------------------------------------------------
# Repetition score (char 20-gram) -- identical to bec-ocr-training's
# ``analyze_rep_bands.rep_score`` / loop-hallucination docs.
# ---------------------------------------------------------------------------


def rep_score(s: str, n: int = 20) -> float:
    if not s or len(s) < n + 1:
        return 0.0
    grams = [s[i : i + n] for i in range(len(s) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# Script / mixed-script content signals.
# ---------------------------------------------------------------------------

TIBETAN_START = 0x0F00
TIBETAN_END = 0x0FFF

# Codepoints strongly associated with Sanskrit transliterated in Tibetan script
# (retroflex bases + subjoined, visarga/anusvara/candrabindu, achung/ā length,
# tsa-phru, fixed-form subjoined ya/ra/wa/ha used in conjuncts, etc.).
def is_tibetan_only(text: str) -> bool:
    """True when text is only Tibetan-block codepoints and ASCII spaces."""
    return all(ch == " " or (TIBETAN_START <= ord(ch) <= TIBETAN_END) for ch in text)


def non_tibetan_char_count(text: str) -> int:
    return sum(
        1
        for ch in text
        if ch not in (" ", "\n") and not (TIBETAN_START <= ord(ch) <= TIBETAN_END)
    )


def strip_to_tibetan(text: str) -> str:
    """Keep only Tibetan-block codepoints and single ASCII spaces.

    Guarantees the rendered text (and therefore the ground truth) contains no
    non-Tibetan-Unicode characters (Latin digits, punctuation from other
    scripts, etc.). Runs of removed characters collapse to a single space.
    """
    out = [ch if (TIBETAN_START <= ord(ch) <= TIBETAN_END) else " " for ch in text]
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


# ---------------------------------------------------------------------------
# LuaLaTeX / TeX helpers (adapted from synthetic_benchmark/render_batches.py so
# ground truth is reconstructed line-for-line exactly as in the main pipeline).
# ---------------------------------------------------------------------------


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "")
        .replace("%", "")
        .replace("$", "")
        .replace("&", "")
        .replace("#", "")
        .replace("_", "")
        .replace("{", "")
        .replace("}", "")
    )


def add_marked_breakpoints(text: str) -> str:
    text = re.sub("་(?!།)", lambda _: "་\\syllmark\\allowbreak{}", text)
    text = text.replace(" ", "\\syllmark\\hspace{0.35em}\\allowbreak{}")
    text = text.replace("\\allowbreak{}", "\\allowbreak{}%\n")
    return text + r"\syllmark{}"


def split_marker_units(text: str) -> list[str]:
    units: list[str] = []
    buf: list[str] = []
    for i, ch in enumerate(text):
        buf.append(ch)
        next_ch = text[i + 1] if i + 1 < len(text) else ""
        if ch == TSHEG and next_ch != SHAD:
            units.append("".join(buf))
            buf = []
        elif ch == " ":
            units.append("".join(buf))
            buf = []
    if buf:
        units.append("".join(buf))
    return units


def fmt_dim(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _lua_page_map_code(page_map_path: Path, line_map_path: Path) -> str:
    return rf"""\directlua{{
synthetic_current_chunk = ""
synthetic_page_no = 0
synthetic_page_map_path = {json.dumps(str(page_map_path))}
synthetic_line_map_path = {json.dumps(str(line_map_path))}
local node = node
local hlist_id = node.id("hlist")
local vlist_id = node.id("vlist")
local kern_id = node.id("kern")
local glyph_id = node.id("glyph")

local function child_list(n)
  local child = n.head
  if type(child) == "number" and node.getlist then
    child = node.getlist(n)
  end
  if type(child) == "number" then
    return nil
  end
  return child
end

local function has_glyph(head)
  if not head or type(head) == "number" then
    return false
  end
  for n in node.traverse(head) do
    if n.id == glyph_id then
      return true
    elseif n.id == hlist_id or n.id == vlist_id then
      local child = child_list(n)
      if child and has_glyph(child) then
        return true
      end
    end
  end
  return false
end

local function is_text_hlist(n)
  local child = child_list(n)
  return child and has_glyph(child)
end

local function count_text_lines(head)
  local count = 0
  local function walk(list)
    if not list or type(list) == "number" then
      return
    end
    for n in node.traverse(list) do
      if n.id == hlist_id then
        if is_text_hlist(n) then
          count = count + 1
        end
      elseif n.id == vlist_id then
        local child = child_list(n)
        if child then walk(child) end
      end
    end
  end
  walk(head)
  return count
end

local function collect_markers_in_hlist(h)
  local ids = {{}}
  local child = child_list(h)
  if not child or type(child) == "number" then
    return ids
  end
  for n in node.traverse(child) do
    if n.id == kern_id then
      local v = node.get_attribute(n, tex.attribute.syllattr)
      if v then table.insert(ids, v) end
    end
  end
  return ids
end

local function collect_line_markers(head, lines)
  if not head or type(head) == "number" then
    return
  end
  for n in node.traverse(head) do
    if n.id == hlist_id then
      local ids = collect_markers_in_hlist(n)
      if type(ids) == "table" and next(ids) then
        table.insert(lines, ids)
      end
    elseif n.id == vlist_id then
      local child = child_list(n)
      if child then collect_line_markers(child, lines) end
    end
  end
end

local function expand_line_ink_extents(head)
  if not head or type(head) == "number" then
    return head
  end
  for line in node.traverse(head) do
    if line.id == hlist_id and is_text_hlist(line) then
      local maximum_upward_offset = 0
      local maximum_downward_offset = 0
      local child = child_list(line)
      if child then
        for glyph in node.traverse_id(glyph_id, child) do
          local yoffset = glyph.yoffset or 0
          maximum_upward_offset = math.max(maximum_upward_offset, yoffset)
          maximum_downward_offset = math.max(maximum_downward_offset, -yoffset)
        end
      end
      local baseline = tex.baselineskip and tex.baselineskip.width or 0
      local ordinary_offset = baseline
      local maximum_extra = baseline * 0.75
      local extra_height = math.min(
        maximum_extra,
        math.max(0, maximum_upward_offset - ordinary_offset)
      )
      local extra_depth = math.min(
        maximum_extra,
        math.max(0, maximum_downward_offset - ordinary_offset)
      )
      line.height = (line.height or 0) + extra_height
      line.depth = (line.depth or 0) + extra_depth
    end
  end
  return head
end

function synthetic_set_chunk(id)
  synthetic_current_chunk = id
  tex.count.syllid = 0
end

local function synthetic_pre_shipout_filter(head)
  synthetic_page_no = synthetic_page_no + 1
  local mode = synthetic_page_no == 1 and "w" or "a"
  local f = io.open(synthetic_page_map_path, mode)
  if f then
    f:write(synthetic_page_no .. "," .. synthetic_current_chunk .. "," .. count_text_lines(head) .. string.char(10))
    f:close()
  end
  local lines = {{}}
  local ok, err = pcall(function() collect_line_markers(head, lines) end)
  if not ok then
    lines = {{}}
  end
  local lf = io.open(synthetic_line_map_path, synthetic_page_no == 1 and "w" or "a")
  if lf then
    for i, ids in ipairs(lines) do
      lf:write(synthetic_page_no .. "," .. synthetic_current_chunk .. "," .. i .. ",")
      if type(ids) == "table" then
        for j, marker_id in ipairs(ids) do
          if j > 1 then lf:write(" ") end
          lf:write(marker_id)
        end
      elseif ids then
        lf:write(ids)
      end
      lf:write(string.char(10))
    end
    lf:close()
  end
  return head
end

if luatexbase and luatexbase.add_to_callback then
  luatexbase.add_to_callback("post_linebreak_filter", expand_line_ink_extents, "synthetic_line_extents")
  luatexbase.add_to_callback("pre_shipout_filter", synthetic_pre_shipout_filter, "synthetic_page_map")
else
  callback.register("post_linebreak_filter", expand_line_ink_extents)
  callback.register("pre_shipout_filter", synthetic_pre_shipout_filter)
end
}}"""


def _fontspec_options(font_abs_path: Path, face_index: str) -> str:
    parts = [
        f"Path={font_abs_path.parent.as_posix()}/",
        f"UprightFont={{{font_abs_path.name}}}",
        "Renderer=HarfBuzz",
        "RawFeature={script=tibt}",
    ]
    if face_index:
        parts.append(f"FontIndex={face_index}")
    return ",\n  ".join(parts)


def make_tex(
    *,
    render_id: str,
    text: str,
    font_abs_path: Path,
    face_index: str,
    font_size_pt: float,
    line_spacing_factor: float,
    page_w_mm: float,
    page_h_mm: float,
    margin_x_mm: float,
    margin_y_mm: float,
    clearance_pt: float,
    page_map_path: Path,
) -> str:
    baseline = font_size_pt * line_spacing_factor
    body = add_marked_breakpoints(tex_escape(text))
    page = (
        "\\noindent\n"
        f"\\directlua{{synthetic_set_chunk(\"{render_id}\")}}\n"
        f"\\fontsize{{{fmt_dim(font_size_pt)}pt}}{{{fmt_dim(baseline)}pt}}\\selectfont\n"
        f"{body}\n"
        "\\clearpage\n"
    )
    return rf"""\documentclass[12pt]{{article}}
\pagestyle{{empty}}
\usepackage{{fontspec}}
\usepackage[
  paperwidth={fmt_dim(page_w_mm)}mm,
  paperheight={fmt_dim(page_h_mm)}mm,
  left={fmt_dim(margin_x_mm)}mm,
  right={fmt_dim(margin_x_mm)}mm,
  top={fmt_dim(margin_y_mm)}mm,
  bottom={fmt_dim(margin_y_mm)}mm
]{{geometry}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0pt}}
\lineskiplimit=0.5pt
\lineskip={fmt_dim(clearance_pt)}pt
\emergencystretch=1em
\tolerance=9999
\pretolerance=9999
\newcount\syllid
\syllid=0
\newattribute\syllattr
\def\syllmark{{%
  \global\advance\syllid by 1
  \directlua{{
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
  }}%
}}
{_lua_page_map_code(page_map_path, page_map_path.with_suffix(".lines.csv"))}
\setmainfont{{DUMMY}}[
  {_fontspec_options(font_abs_path, face_index)}
]
\begin{{document}}
{page}\end{{document}}
"""


def parse_page_map(path: Path) -> list[dict[str, int]]:
    pages: list[dict[str, int]] = []
    if not path.exists():
        return pages
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, fieldnames=["physical_page", "render_id", "line_count"]):
            if not row["render_id"]:
                continue
            pages.append(
                {
                    "physical_page": int(row["physical_page"]),
                    "line_count": int(row["line_count"]),
                }
            )
    return pages


def parse_line_map(path: Path) -> dict[int, list[list[int]]]:
    lines_by_page: dict[int, list[list[int]]] = defaultdict(list)
    if not path.exists():
        return lines_by_page
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(
            f, fieldnames=["physical_page", "render_id", "line_index", "marker_ids"]
        ):
            if not row["render_id"]:
                continue
            marker_ids = [
                int(item)
                for item in str(row.get("marker_ids") or "").split()
                if item.strip().isdigit()
            ]
            lines_by_page[int(row["physical_page"])].append(marker_ids)
    return lines_by_page


def rebuild_transcription(text: str, page_no: int, lines_by_page: dict[int, list[list[int]]]) -> str:
    """Reconstruct the rendered page text with newlines from TeX line markers."""
    line_marker_ids = lines_by_page.get(page_no, [])
    units = split_marker_units(text)
    if not line_marker_ids:
        return text.strip()
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
    joined = "\n".join(lines).strip()
    return joined or text.strip()


# ---------------------------------------------------------------------------
# Minimal mandatory raster naturalization (production "profile A").
#
# Vendored from synthetic_benchmark/document_augmentation.py. This is the least
# processing the production renderer ever applies to a page; we apply *only*
# this and nothing else, so no augmentation-difficulty variance leaks into the
# axis sweeps. Depends on cv2 + numpy only.
# ---------------------------------------------------------------------------


def _naturalization_params_profile_a(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    blur = rng.uniform(0.35, 0.65)
    dropout = rng.uniform(0.0300, 0.0600)
    noise = rng.uniform(0.60, 1.20)
    displacement = rng.uniform(0.60, 1.20)
    return {
        "blur_sigma": round(blur, 6),
        "white_dropout_rate": round(dropout, 8),
        "luminance_noise_sigma": round(noise, 6),
        "displacement_px": round(displacement, 6),
    }


def _apply_local_micro_distortion(image, displacement_px, rng):
    import cv2
    import numpy as np

    if displacement_px <= 0:
        return image
    height, width = image.shape[:2]
    grid_h = max(3, int(np.ceil(height / 32)) + 2)
    grid_w = max(3, int(np.ceil(width / 32)) + 2)
    dx = cv2.resize(rng.normal(size=(grid_h, grid_w)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.normal(size=(grid_h, grid_w)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    rms = max(float(np.sqrt((np.square(dx).mean() + np.square(dy).mean()) / 2.0)), 1e-6)
    limit = 2.5 * displacement_px
    dx = np.clip(dx * displacement_px / rms, -limit, limit)
    dy = np.clip(dy * displacement_px / rms, -limit, limit)
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(image, x + dx, y + dy, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _apply_white_ink_dropout(image, rate, rng):
    import cv2
    import numpy as np

    if rate <= 0:
        return image
    grayscale = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_background = cv2.morphologyEx(
        grayscale, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    ink_mask = (local_background.astype(np.int16) - grayscale.astype(np.int16) >= 18) & (grayscale < 225)
    candidates = np.flatnonzero(ink_mask)
    count = min(len(candidates), max(0, round(len(candidates) * rate)))
    if count == 0:
        return image
    selected = rng.choice(candidates, size=count, replace=False)
    alpha = rng.uniform(0.55, 1.0, size=count).astype(np.float32)
    alpha[rng.random(count) < 0.30] = 1.0
    output = image.astype(np.float32, copy=True)
    flattened = output.reshape(-1)
    flattened[selected] += (255.0 - flattened[selected]) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_raster_naturalization(gray_image, params: dict[str, float], seed: int):
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    out = _apply_local_micro_distortion(gray_image, params["displacement_px"], rng)
    out = _apply_white_ink_dropout(out, params["white_dropout_rate"], rng)
    out = cv2.GaussianBlur(out, (0, 0), params["blur_sigma"])
    noise = rng.normal(0.0, params["luminance_noise_sigma"], size=out.shape[:2]).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Font catalog (minimal loader for a single chosen font).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Font:
    basename: str
    ps_name: str
    font_path: str
    ttc_face_index: str
    font_size_pt: float
    skt_ok: str

    @property
    def abs_path(self) -> Path:
        return BENCHMARK_DIR / self.font_path


def load_font(fonts_csv: Path, basename: str) -> Font:
    with fonts_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("basename", "").strip() == basename:
                return Font(
                    basename=basename,
                    ps_name=row.get("ps_name", "").strip(),
                    font_path=row.get("font_path", "").strip(),
                    ttc_face_index=row.get("ttc_face_index", "").strip(),
                    font_size_pt=float(row.get("font_size_pt") or 24.0),
                    skt_ok=row.get("skt_ok", "").strip(),
                )
    raise SystemExit(f"Font basename {basename!r} not found in {fonts_csv}")


# Tibetan punctuation / digits that may appear outside stacks. Everything else
# in the Tibetan block (letters, subjoined letters, vowels, annotation marks)
# must be accounted for by a coverage-supported stack. This catches orphan
# U+0F35 / U+0F37 that botok's stack tokenizer drops between spaces/shads —
# the exact leak that let ``axis_stack_rarity_bin0_0004.jpg`` through.
_STACK_EXEMPT_CHARS = frozenset(
    chr(cp)
    for a, b in (
        (0x0F04, 0x0F14),  # head marks, shad variants, tsheg
        (0x0F20, 0x0F29),  # digits
        (0x0F3A, 0x0F3D),  # brackets
    )
    for cp in range(a, b + 1)
) | frozenset(
    chr(cp)
    for cp in (
        0x0F00, 0x0F01, 0x0F02, 0x0F03,
        0x0F34, 0x0F36, 0x0F38,
        0x0F3E, 0x0F3F, 0x0F85,
        0x0FD0, 0x0FD1, 0x0FD2, 0x0FD3, 0x0FD4,
    )
)


def text_stacks_supported(text: str, supported_stacks: set[str]) -> bool:
    """True when every Tibetan stack in ``text`` is supported for the font.

    Uses the same stack tokenization and ``stack_support.parquet`` gate as the
    main render planner: a page is accepted only when every stack shapes cleanly
    under the coverage_report HarfBuzz + placement rules (``ok`` and zero
    placement warnings).

    Additionally rejects orphan letters/marks that botok does not emit as stacks
    (lone U+0F35 / U+0F37 between spaces are the motivating case): every
    non-punctuation Tibetan codepoint in the page must appear inside some
    tokenized stack, and every such stack must be coverage-supported.
    """
    stacks = tokenize_tibetan_stacks(text)
    if not stacks:
        return False
    if not all(stack in supported_stacks for stack in stacks):
        return False
    from collections import Counter

    covered: Counter[str] = Counter()
    for stack in stacks:
        covered.update(stack)
    body = Counter(
        ch
        for ch in text
        if TIBETAN_START <= ord(ch) <= TIBETAN_END and ch not in _STACK_EXEMPT_CHARS
    )
    leftover = body - covered
    return not leftover


# ---------------------------------------------------------------------------
# Source-text units.
# ---------------------------------------------------------------------------


def split_shad_units(text: str) -> list[str]:
    """Split normalized page text into shad-delimited units (keep the shad)."""
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch == SHAD:
            unit = "".join(buf).strip()
            if unit:
                parts.append(unit)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def prefix_to_chars(text: str, target_chars: int) -> str:
    """Take a whole-unit prefix of `text` whose length is closest to target."""
    units = split_shad_units(text)
    out: list[str] = []
    total = 0
    for unit in units:
        add = len(unit) + (1 if out else 0)
        if out and total + add > target_chars:
            # accept overshoot only if we are still far below target
            if total >= target_chars * 0.75:
                break
        out.append(unit)
        total += add
        if total >= target_chars:
            break
    return " ".join(out).strip() or text[:target_chars]


_SYLLABLE_SPLIT_RE = re.compile(r"[་།\s]+")


def count_syllables(text: str) -> int:
    """Count Tibetan syllables (tsheg/shad/whitespace-delimited bodies)."""
    parts = _SYLLABLE_SPLIT_RE.split(text.replace("\n", " "))
    return sum(
        1
        for part in parts
        if part and any(TIBETAN_START <= ord(ch) <= TIBETAN_END for ch in part)
    )


def prefix_to_syllables(text: str, target_syllables: int) -> str:
    """Take a prefix of ``text`` with about ``target_syllables`` syllables.

    Cuts on an existing syllable boundary (after tsheg/shad/whitespace).
    Preserves the original characters as-is — does not re-tokenize or insert
    separators between syllables.
    """
    if target_syllables <= 0:
        return ""
    n = 0
    in_syll = False
    cut = 0
    for i, ch in enumerate(text):
        is_boundary = ch in (TSHEG, SHAD) or ch.isspace()
        if is_boundary:
            if in_syll:
                n += 1
                in_syll = False
                cut = i + 1
                if n >= target_syllables:
                    break
            elif ch in (TSHEG, SHAD):
                cut = i + 1
        elif TIBETAN_START <= ord(ch) <= TIBETAN_END:
            in_syll = True
    else:
        if in_syll:
            n += 1
        cut = len(text)
    return text[:cut].strip()


# ---------------------------------------------------------------------------
# Rendering a single page.
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    font_abs_path: str
    face_index: str
    page_w_mm: float
    margin_x_mm: float
    margin_y_mm: float
    line_spacing_factor: float
    clearance_pt: float
    image_width_px: int
    jpeg_quality: int
    lualatex: str = "lualatex"
    pdftoppm: str = "pdftoppm"


def _line_height_mm(font_size_pt: float, line_spacing_factor: float) -> float:
    return font_size_pt * line_spacing_factor / 72.27 * 25.4


def render_page(
    *,
    text: str,
    font_size_pt: float,
    cfg: RenderConfig,
    work_dir: Path,
    naturalization_seed: int,
) -> dict:
    """Render one page and return {image_path, transcription, line_count, ...}.

    Sizes the page height to the rendered line count (tight), retrying with more
    vertical room if the text overflows one physical page.
    """
    import cv2

    # Absolute paths: LuaLaTeX runs with cwd=work_dir, so a relative input path
    # (from a relative --out-dir) would be resolved against work_dir and fail.
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    render_id = "r1"
    tex_path = work_dir / "page.tex"
    pdf_path = work_dir / "page.pdf"
    page_map_path = work_dir / "page.pages.csv"
    line_map_path = page_map_path.with_suffix(".lines.csv")

    line_h = _line_height_mm(font_size_pt, cfg.line_spacing_factor)
    # First estimate: assume ~ this many chars per line so the initial height is
    # close; overflow triggers a retry with more room, so the estimate is only a
    # starting point.
    est_chars_per_line = max(8.0, cfg.image_width_px / 120.0)
    est_lines = max(1, math.ceil(len(text) / est_chars_per_line)) + 2

    result = None
    for attempt in range(6):
        page_h_mm = 2 * cfg.margin_y_mm + est_lines * line_h + line_h
        for p in (page_map_path, line_map_path, pdf_path):
            if p.exists():
                p.unlink()
        tex_path.write_text(
            make_tex(
                render_id=render_id,
                text=text,
                font_abs_path=Path(cfg.font_abs_path),
                face_index=cfg.face_index,
                font_size_pt=font_size_pt,
                line_spacing_factor=cfg.line_spacing_factor,
                page_w_mm=cfg.page_w_mm,
                page_h_mm=page_h_mm,
                margin_x_mm=cfg.margin_x_mm,
                margin_y_mm=cfg.margin_y_mm,
                clearance_pt=cfg.clearance_pt,
                page_map_path=page_map_path,
            ),
            encoding="utf-8",
        )
        log_path = work_dir / f"lualatex.{attempt}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [
                    cfg.lualatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    f"-output-directory={work_dir}",
                    str(tex_path),
                ],
                cwd=work_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0 or not pdf_path.exists() or not page_map_path.exists():
            raise RuntimeError(f"lualatex failed (see {log_path})")
        pages = parse_page_map(page_map_path)
        content_pages = [p for p in pages if p["line_count"] > 0]
        n_physical = len({p["physical_page"] for p in pages})
        if n_physical > 1:
            # overflowed to a second page -> grow and retry
            est_lines = int(est_lines * 1.4) + 2
            continue
        lines_by_page = parse_line_map(line_map_path)
        line_count = content_pages[0]["line_count"] if content_pages else 0
        # Re-fit height snugly to the realized line count if our estimate was off
        target_lines = max(1, line_count)
        snug_lines = target_lines + 1
        if abs(snug_lines - est_lines) > 1 and attempt < 4:
            est_lines = snug_lines
            # one more render at the snug height for a tight page
            page_h_mm = 2 * cfg.margin_y_mm + est_lines * line_h + line_h
            for p in (page_map_path, line_map_path, pdf_path):
                if p.exists():
                    p.unlink()
            tex_path.write_text(
                make_tex(
                    render_id=render_id,
                    text=text,
                    font_abs_path=Path(cfg.font_abs_path),
                    face_index=cfg.face_index,
                    font_size_pt=font_size_pt,
                    line_spacing_factor=cfg.line_spacing_factor,
                    page_w_mm=cfg.page_w_mm,
                    page_h_mm=page_h_mm,
                    margin_x_mm=cfg.margin_x_mm,
                    margin_y_mm=cfg.margin_y_mm,
                    clearance_pt=cfg.clearance_pt,
                    page_map_path=page_map_path,
                ),
                encoding="utf-8",
            )
            with log_path.open("a", encoding="utf-8") as log:
                proc = subprocess.run(
                    [
                        cfg.lualatex,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-file-line-error",
                        f"-output-directory={work_dir}",
                        str(tex_path),
                    ],
                    cwd=work_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if proc.returncode != 0 or not pdf_path.exists():
                raise RuntimeError(f"lualatex re-fit failed (see {log_path})")
            pages = parse_page_map(page_map_path)
            if len({p["physical_page"] for p in pages}) > 1:
                est_lines = int(est_lines * 1.4) + 2
                continue
            lines_by_page = parse_line_map(line_map_path)
            content_pages = [p for p in pages if p["line_count"] > 0]
            line_count = content_pages[0]["line_count"] if content_pages else line_count
        first_page_no = content_pages[0]["physical_page"] if content_pages else 1
        transcription = rebuild_transcription(text, first_page_no, lines_by_page)
        result = {"pdf_path": pdf_path, "line_count": line_count, "transcription": transcription}
        break
    if result is None:
        raise RuntimeError("could not fit page on a single physical page")

    # Rasterize with poppler at the fixed pixel width, grayscale.
    ppm_prefix = work_dir / "raster"
    log_path = work_dir / "pdftoppm.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [
                cfg.pdftoppm,
                "-f", "1", "-l", "1",
                "-jpeg", "-jpegopt", "quality=95",
                "-gray",
                "-scale-to-x", str(cfg.image_width_px),
                "-scale-to-y", "-1",
                str(result["pdf_path"]),
                str(ppm_prefix),
            ],
            cwd=work_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"pdftoppm failed (see {log_path})")
    candidates = sorted(work_dir.glob("raster-*.jpg"))
    if not candidates:
        raise RuntimeError("pdftoppm produced no image")
    raster_path = candidates[0]

    params = _naturalization_params_profile_a(naturalization_seed)
    image = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"could not read raster {raster_path}")
    height_px = image.shape[0]
    natural = apply_raster_naturalization(image, params, naturalization_seed)
    from PIL import Image

    final_path = work_dir / "final.jpg"
    Image.fromarray(natural).convert("L").save(final_path, format="JPEG", quality=cfg.jpeg_quality)
    return {
        "image_path": final_path,
        "transcription": result["transcription"],
        "line_count": int(result["line_count"]),
        "image_height_px": int(height_px),
        "naturalization": params,
    }


# ---------------------------------------------------------------------------
# Job specification and per-axis plan.
# ---------------------------------------------------------------------------


@dataclass
class Job:
    seq: int
    axis: str
    benchmark_group: str
    axis_bin: int
    axis_value: float
    font_size_pt: float
    text: str
    script: str
    provenance: dict = field(default_factory=dict)


def _stable_seed(*values) -> int:
    payload = ":".join(str(v) for v in values)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Axis definitions (bins).  Values are *targets*; realized metrics are measured
# post-render and written to the manifest.
# ---------------------------------------------------------------------------

# stack rarity: bins by page rare-stack score (0 = all-common stacks; higher =
# rarer distinct stacks per page). Bin 0 is the all-common center. Populations
# in the near-center-length window (full BoCorpus): ~289k / 91k / 17k / 4k / 5k.
RARITY_BANDS = [(0.0, 1e-9), (1e-9, 0.03), (0.03, 0.10), (0.10, 0.22), (0.22, 1.01)]
# within-page repetition: target char-20-gram rep_score.
REPETITION_TARGETS = [0.0, 0.15, 0.30, 0.45, 0.60]
# length: target *syllables* per page (density held constant by fixed width/font).
# Bin 0 starts near a short phrase (~15 syllables). The top bin is the longest
# page in the BEC training corpus ``bec_mixed_elie_v3`` (1005 syllables) + 20%
# → 1206. Intermediate bins are log-spaced.
LENGTH_TARGETS = [15, 45, 135, 400, 1206]
# font size / effective resolution: absolute rendered point sizes.
FONT_SIZE_TARGETS = [12.0, 18.0, 28.0]

AXIS_STACK_RARITY = "stack_rarity"
AXIS_REPETITION = "repetition"
AXIS_LENGTH = "length"
AXIS_FONT_SIZE = "font_size"


# ---------------------------------------------------------------------------
# Page rare-stack scoring.
#
# The chunks parquet stores the per-page set of unique Tibetan stacks (the
# ``stacks`` column). We score how rare a page's stacks are using the same
# corpus-frequency rarity table (``coverage_report/bocorpus_stacks.csv``) and
# the same percentile-clipped self-information the repository uses in
# ``synthetic_common.load_stack_rarity_scores`` (vendored below so this builder
# has no botok dependency). The per-page score is the mean rarity of the 20
# rarest distinct stacks on the page: 0.0 for an all-common page, approaching
# 1.0 for a page dominated by rare / unseen stacks.
# ---------------------------------------------------------------------------

RARITY_IGNORED_MARKS = frozenset({"\u0f35", "\u0f37"})
RARITY_TOP_K = 20


def _is_pure_tibetan_stack(stack: str) -> bool:
    return bool(stack) and all(TIBETAN_START <= ord(ch) <= TIBETAN_END for ch in stack)


def _has_tibetan_letter(stack: str) -> bool:
    import unicodedata

    return any(unicodedata.category(ch).startswith("L") for ch in stack)


def _rarity_equivalent_stack(stack: str) -> str:
    return "".join(ch for ch in stack if ch not in RARITY_IGNORED_MARKS)


def load_stack_rarity_scores(stacks_csv: Path | str) -> dict[str, float]:
    """Map each stack to a [0,1] rarity (vendored from synthetic_common)."""
    raw_to_equivalent: dict[str, str] = {}
    counts: dict[str, int] = {}
    with Path(stacks_csv).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stack = (row.get("stack") or "").strip()
            if not stack or not _is_pure_tibetan_stack(stack) or not _has_tibetan_letter(stack):
                continue
            raw = row.get("nb_occurences") or row.get("nb_occurrences") or "0"
            try:
                count = int(str(raw).strip() or "0")
            except ValueError:
                count = 0
            if count > 0:
                equivalent = _rarity_equivalent_stack(stack)
                raw_to_equivalent[stack] = equivalent
                counts[equivalent] = counts.get(equivalent, 0) + count
    if not counts:
        return {}
    total = sum(counts.values())
    infos = sorted(-math.log(count / total) for count in counts.values())
    lo = infos[int(0.05 * (len(infos) - 1))]
    hi = infos[int(0.95 * (len(infos) - 1))]
    scale = hi - lo
    if scale <= 0:
        equivalent_scores = {stack: 0.0 for stack in counts}
    else:
        equivalent_scores = {
            stack: min(1.0, max(0.0, ((-math.log(count / total)) - lo) / scale))
            for stack, count in counts.items()
        }
    return equivalent_scores | {
        stack: equivalent_scores[equivalent] for stack, equivalent in raw_to_equivalent.items()
    }


def page_rarity(stacks_field: str, rarity_by_stack: dict[str, float]) -> float:
    """Mean rarity of the 20 rarest distinct Tibetan stacks on the page."""
    stacks = [
        s
        for s in (stacks_field or "").split()
        if s and _is_pure_tibetan_stack(s) and _has_tibetan_letter(s)
    ]
    if not stacks:
        return 0.0
    rarities = [
        rarity_by_stack.get(s, rarity_by_stack.get(_rarity_equivalent_stack(s), 1.0))
        for s in stacks
    ]
    top = sorted(rarities, reverse=True)[: RARITY_TOP_K]
    return sum(top) / len(top)


@dataclass
class Candidate:
    chunk_id: str
    bocorpus_row: int
    char_start: int
    char_end: int
    text: str
    char_count: int
    rarity: float


def load_candidates(
    chunks_parquet: Path,
    *,
    scan_limit: int,
    seed: int,
    stacks_csv: Path,
) -> list[Candidate]:
    import pyarrow.parquet as pq

    rarity_by_stack = load_stack_rarity_scores(stacks_csv)
    if not rarity_by_stack:
        raise SystemExit(f"Empty rarity table from {stacks_csv}")

    cols = ["chunk_id", "bocorpus_row", "char_start", "char_end", "text", "char_count", "stacks"]
    table = pq.read_table(chunks_parquet, columns=cols)
    n = table.num_rows
    order = list(range(n))
    random.Random(seed).shuffle(order)
    chunk_id = table.column("chunk_id").to_pylist()
    boc = table.column("bocorpus_row").to_pylist()
    cs = table.column("char_start").to_pylist()
    ce = table.column("char_end").to_pylist()
    txt = table.column("text").to_pylist()
    cc = table.column("char_count").to_pylist()
    stacks = table.column("stacks").to_pylist()
    out: list[Candidate] = []
    for idx in order:
        c = int(cc[idx] or 0)
        if c < 200:
            continue
        out.append(
            Candidate(
                chunk_id=str(chunk_id[idx]),
                bocorpus_row=int(boc[idx] or 0),
                char_start=int(cs[idx] or 0),
                char_end=int(ce[idx] or 0),
                text=str(txt[idx] or ""),
                char_count=c,
                rarity=page_rarity(str(stacks[idx] or ""), rarity_by_stack),
            )
        )
        if len(out) >= scan_limit:
            break
    return out


def build_jobs(
    candidates: list[Candidate],
    *,
    supported_stacks: set[str],
    per_bin: int,
    font_per_bin: int,
    center_chars: int,
    center_font_size_pt: float,
    script: str,
    seed: int,
) -> list[Job]:
    used: set[str] = set()
    jobs: list[Job] = []
    seq = 0

    # center length band for the rarity axis: hold length ~ center while using
    # the whole-chunk stacks (so the page rarity score is exact for the render).
    center_lo = center_chars
    center_hi = center_chars + 400

    def next_seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    def supported(text: str) -> bool:
        return text_stacks_supported(text, supported_stacks)

    def is_clean_common(cand: Candidate) -> bool:
        """Clean common-stack, Tibetan-only, long-enough base chunk."""
        return (
            cand.chunk_id not in used
            and cand.rarity < 0.02
            and cand.char_count >= center_chars
            and is_tibetan_only(cand.text)
        )

    # ---- Axis: stack rarity -------------------------------------------------
    # Whole-chunk pages near the center length, binned by page rare-stack score.
    band_hits: dict[int, int] = defaultdict(int)
    for cand in candidates:
        if all(band_hits[b] >= per_bin for b in range(len(RARITY_BANDS))):
            break
        if cand.chunk_id in used or not (center_lo <= cand.char_count <= center_hi):
            continue
        band = None
        for bi, (lo, hi) in enumerate(RARITY_BANDS):
            if lo <= cand.rarity < hi:
                band = bi
                break
        if band is None or band_hits[band] >= per_bin:
            continue
        if not is_tibetan_only(cand.text):
            continue
        prefix = prefix_to_chars(cand.text, center_chars)
        if rep_score(prefix) >= 0.15 or not supported(prefix):
            continue
        used.add(cand.chunk_id)
        band_hits[band] += 1
        jobs.append(
            Job(
                seq=next_seq(),
                axis=AXIS_STACK_RARITY,
                benchmark_group=f"axis_{AXIS_STACK_RARITY}",
                axis_bin=band,
                axis_value=round(cand.rarity, 6),
                font_size_pt=center_font_size_pt,
                text=prefix,
                script=script,
                provenance={"chunk_id": cand.chunk_id, "bocorpus_row": cand.bocorpus_row,
                            "char_start": cand.char_start, "char_end": cand.char_end,
                            "construction": "chunk_prefix",
                            "page_rarity_score": round(cand.rarity, 6)},
            )
        )
    for b, (lo, hi) in enumerate(RARITY_BANDS):
        if band_hits[b] < per_bin:
            print(f"WARNING: stack_rarity bin {b} [{lo:.3g},{hi:.3g}) only "
                  f"{band_hits[b]}/{per_bin} (increase --scan-limit)")

    # ---- Shared common-base pool -------------------------------------------
    # A length job may consume several bases (long pages are built by
    # concatenating consecutive common pages), so budget generously. Length
    # targets are in syllables; a center page is ~center_chars/4 syllables.
    center_syllables = max(1, center_chars // 4)
    length_bases_per_job = sum(
        max(1, math.ceil(t / center_syllables)) for t in LENGTH_TARGETS
    )
    need_common = (
        len(REPETITION_TARGETS) * per_bin
        + length_bases_per_job * per_bin * 2  # retries when a concat fails coverage
        + len(FONT_SIZE_TARGETS) * font_per_bin
        + 40
    )
    common_bases: list[Candidate] = []
    for cand in candidates:
        if len(common_bases) >= need_common:
            break
        if cand.chunk_id in used:
            continue
        if is_clean_common(cand):
            prefix = prefix_to_chars(cand.text, center_chars)
            if rep_score(prefix) < 0.12 and supported(prefix):
                common_bases.append(cand)
                used.add(cand.chunk_id)

    pool = iter(common_bases)

    def pull_base(axis_name: str, bi: int) -> Candidate | None:
        try:
            return next(pool)
        except StopIteration:
            print(f"WARNING: {axis_name} axis short of common base chunks at bin {bi} "
                  f"(have {len(common_bases)}, increase --scan-limit)")
            return None

    # ---- Axis: repetition ---------------------------------------------------
    for bi, target in enumerate(REPETITION_TARGETS):
        for _ in range(per_bin):
            cand = pull_base(AXIS_REPETITION, bi)
            if cand is None:
                break
            prefix = prefix_to_chars(cand.text, center_chars)
            text = build_repetition_text(prefix, target, center_chars)
            if not supported(text):
                continue
            jobs.append(
                Job(
                    seq=next_seq(),
                    axis=AXIS_REPETITION,
                    benchmark_group=f"axis_{AXIS_REPETITION}",
                    axis_bin=bi,
                    axis_value=round(target, 6),
                    font_size_pt=center_font_size_pt,
                    text=text,
                    script=script,
                    provenance={"chunk_id": cand.chunk_id, "bocorpus_row": cand.bocorpus_row,
                                "char_start": cand.char_start, "char_end": cand.char_end,
                                "construction": "synthetic_repetition",
                                "target_rep_score": target,
                                "realized_rep_score_pretex": round(rep_score(text), 6)},
                )
            )

    # ---- Axis: length -------------------------------------------------------
    # Density (chars/line) is held constant by the fixed font size and page
    # width; longer pages are built by concatenating consecutive common pages
    # and trimming to the target syllable count, so the page simply gets
    # taller at the same pixel width and dpi. Targets are syllables, not chars.
    for bi, target in enumerate(LENGTH_TARGETS):
        filled = 0
        attempts = 0
        while filled < per_bin and attempts < per_bin * 4:
            attempts += 1
            first: Candidate | None = None
            source_ids: list[str] = []
            buf = ""
            while count_syllables(buf) < target:
                cand = pull_base(AXIS_LENGTH, bi)
                if cand is None:
                    break
                if first is None:
                    first = cand
                source_ids.append(cand.chunk_id)
                buf = cand.text if not buf else (buf + " " + cand.text)
            if first is None:
                break
            text = prefix_to_syllables(buf, target)
            if not supported(text):
                continue
            filled += 1
            jobs.append(
                Job(
                    seq=next_seq(),
                    axis=AXIS_LENGTH,
                    benchmark_group=f"axis_{AXIS_LENGTH}",
                    axis_bin=bi,
                    axis_value=float(target),
                    font_size_pt=center_font_size_pt,
                    text=text,
                    script=script,
                    provenance={"chunk_id": first.chunk_id, "bocorpus_row": first.bocorpus_row,
                                "char_start": first.char_start, "char_end": first.char_end,
                                "construction": "concat_chunk_prefix" if len(source_ids) > 1 else "chunk_prefix",
                                "source_chunk_ids": ",".join(source_ids),
                                "target_syllables": target,
                                "realized_syllables_pretex": count_syllables(text),
                                "realized_chars_pretex": len(text)},
                )
            )
        if filled < per_bin:
            print(f"WARNING: length bin {bi} target={target} only "
                  f"{filled}/{per_bin} (increase --scan-limit)")

    # ---- Axis: font size / resolution --------------------------------------
    for bi, size in enumerate(FONT_SIZE_TARGETS):
        for _ in range(font_per_bin):
            cand = pull_base(AXIS_FONT_SIZE, bi)
            if cand is None:
                break
            prefix = prefix_to_chars(cand.text, center_chars)
            if not supported(prefix):
                continue
            jobs.append(
                Job(
                    seq=next_seq(),
                    axis=AXIS_FONT_SIZE,
                    benchmark_group=f"axis_{AXIS_FONT_SIZE}",
                    axis_bin=bi,
                    axis_value=float(size),
                    font_size_pt=float(size),
                    text=prefix,
                    script=script,
                    provenance={"chunk_id": cand.chunk_id, "bocorpus_row": cand.bocorpus_row,
                                "char_start": cand.char_start, "char_end": cand.char_end,
                                "construction": "chunk_prefix", "target_font_size_pt": size},
                )
            )

    return jobs


def build_repetition_text(base_prefix: str, target_rep: float, target_chars: int) -> str:
    """Build litany/mantra-like text with a controlled char-20-gram rep_score.

    Interleaves a repeated motif (a shad-delimited unit) with unique base units
    to approach `target_rep` while keeping the length near `target_chars`.
    """
    units = [u for u in split_shad_units(base_prefix) if u]
    if not units:
        return base_prefix
    if target_rep <= 0.0:
        return base_prefix
    # pick a motif long enough that its 20-grams recur when repeated
    motif = max(units, key=len)
    if len(motif) < 24 and len(units) >= 2:
        ordered = sorted(units, key=len, reverse=True)
        motif = (ordered[0] + " " + ordered[1]).strip()
    unique_units = [u for u in units if u != motif]
    m = len(motif) + 1

    def build(rep_count: int) -> str:
        motif_chars = rep_count * m
        unique_budget = max(0, target_chars - motif_chars)
        chosen: list[str] = []
        total = 0
        for u in unique_units:
            if total >= unique_budget:
                break
            chosen.append(u)
            total += len(u) + 1
        # interleave for a natural litany look
        seq: list[str] = []
        ci = 0
        for _ in range(rep_count):
            seq.append(motif)
            if ci < len(chosen):
                seq.append(chosen[ci])
                ci += 1
        seq.extend(chosen[ci:])
        return " ".join(seq).strip()

    best_text = base_prefix
    best_gap = 1e9
    for rep_count in range(1, 500):
        text = build(rep_count)
        if len(text) > target_chars * 1.6:
            break
        gap = abs(rep_score(text) - target_rep)
        if gap < best_gap:
            best_gap = gap
            best_text = text
        if rep_score(text) >= target_rep + 0.05 and len(text) >= target_chars * 0.8:
            break
    return best_text


# ---------------------------------------------------------------------------
# Rendering orchestration.
# ---------------------------------------------------------------------------

_WORKER_CFG: RenderConfig | None = None
_WORKER_WORKROOT: Path | None = None
_WORKER_SEED: int = 0


def _init_worker(cfg: RenderConfig, workroot: str, seed: int) -> None:
    global _WORKER_CFG, _WORKER_WORKROOT, _WORKER_SEED
    _WORKER_CFG = cfg
    _WORKER_WORKROOT = Path(workroot)
    _WORKER_SEED = seed


def _render_job(job: Job) -> dict:
    assert _WORKER_CFG is not None and _WORKER_WORKROOT is not None
    work_dir = _WORKER_WORKROOT / f"job_{job.seq:04d}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    naturalization_seed = _stable_seed(_WORKER_SEED, "naturalization", job.seq)
    try:
        result = render_page(
            text=job.text,
            font_size_pt=job.font_size_pt,
            cfg=_WORKER_CFG,
            work_dir=work_dir,
            naturalization_seed=naturalization_seed,
        )
        image_bytes = Path(result["image_path"]).read_bytes()
        out = {
            "seq": job.seq,
            "ok": True,
            "image_bytes": image_bytes,
            "transcription": result["transcription"],
            "line_count": result["line_count"],
            "image_height_px": result["image_height_px"],
            "naturalization": result["naturalization"],
        }
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        return out
    except Exception as exc:  # noqa: BLE001
        # Leave the work dir in place on failure so the LuaLaTeX/Poppler log can
        # be inspected.
        return {"seq": job.seq, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def render_all(jobs: list[Job], cfg: RenderConfig, out_dir: Path, *, jobs_parallel: int, seed: int) -> dict[int, dict]:
    workroot = out_dir / "workers"
    workroot.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}
    if jobs_parallel <= 1:
        _init_worker(cfg, str(workroot), seed)
        for i, job in enumerate(jobs, 1):
            res = _render_job(job)
            results[res["seq"]] = res
            status = "ok" if res["ok"] else f"FAIL {res.get('error')}"
            print(f"[{i}/{len(jobs)}] seq={job.seq} {job.axis} bin{job.axis_bin} -> {status}", flush=True)
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=jobs_parallel,
            mp_context=mp.get_context("fork"),
            initializer=_init_worker,
            initargs=(cfg, str(workroot), seed),
        ) as ex:
            futures = {ex.submit(_render_job, job): job for job in jobs}
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                results[res["seq"]] = res
                done += 1
                job = futures[fut]
                status = "ok" if res["ok"] else f"FAIL {res.get('error')}"
                print(f"[{done}/{len(jobs)}] seq={job.seq} {job.axis} bin{job.axis_bin} -> {status}", flush=True)
    # Remove the work root only if every job succeeded; otherwise keep the
    # failing job dirs (with their LuaLaTeX/Poppler logs) for inspection.
    if all(r.get("ok") for r in results.values()):
        shutil.rmtree(workroot, ignore_errors=True)
    return results


# ---------------------------------------------------------------------------
# Output emission.
# ---------------------------------------------------------------------------

PARQUET_COLUMNS = [
    "file_name",
    "transcription",
    "image",
    "script",
    "technology",
    "benchmark_group",
    "axis",
    "axis_bin",
    "axis_value",
]


def emit_outputs(
    jobs: list[Job],
    results: dict[int, dict],
    out_dir: Path,
    *,
    font: Font,
    cfg: RenderConfig,
    config_meta: dict,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    parquet_rows: list[dict] = []
    manifest_rows: list[dict] = []
    n_ok = n_fail = 0
    for job in jobs:
        res = results.get(job.seq)
        if not res or not res.get("ok"):
            n_fail += 1
            print(f"WARNING: dropping seq={job.seq} ({job.axis} bin{job.axis_bin}): "
                  f"{(res or {}).get('error', 'no result')}")
            continue
        n_ok += 1
        transcription = normalize_unicode_text(res["transcription"]).strip()
        file_name = f"{job.benchmark_group}_bin{job.axis_bin}_{job.seq:04d}.jpg"
        (images_dir / file_name).write_bytes(res["image_bytes"])
        parquet_rows.append(
            {
                "file_name": file_name,
                "transcription": transcription,
                "image": res["image_bytes"],
                "script": job.script,
                "technology": "synthetic",
                "benchmark_group": job.benchmark_group,
                "axis": job.axis,
                "axis_bin": int(job.axis_bin),
                "axis_value": float(job.axis_value),
            }
        )
        nat = res.get("naturalization", {})
        manifest_rows.append(
            {
                "file_name": file_name,
                "axis": job.axis,
                "benchmark_group": job.benchmark_group,
                "axis_bin": int(job.axis_bin),
                "axis_value": job.axis_value,
                "script": job.script,
                "font_basename": font.basename,
                "font_ps_name": font.ps_name,
                "font_file": Path(font.font_path).name,
                "font_size_pt": job.font_size_pt,
                "image_width_px": cfg.image_width_px,
                "image_height_px": res.get("image_height_px"),
                "page_width_mm": cfg.page_w_mm,
                "margin_x_mm": cfg.margin_x_mm,
                "margin_y_mm": cfg.margin_y_mm,
                "line_spacing_factor": cfg.line_spacing_factor,
                "jpeg_quality": cfg.jpeg_quality,
                "line_count": res.get("line_count"),
                "char_count_gt": len(transcription),
                "syllable_count_gt": count_syllables(transcription),
                "rep_score_gt": round(rep_score(transcription), 6),
                "non_tibetan_chars_gt": non_tibetan_char_count(transcription),
                "naturalization_profile": "A",
                "naturalization_seed": _stable_seed(config_meta["seed"], "naturalization", job.seq),
                "naturalization_blur_sigma": nat.get("blur_sigma"),
                "naturalization_white_dropout_rate": nat.get("white_dropout_rate"),
                "naturalization_luminance_noise_sigma": nat.get("luminance_noise_sigma"),
                "naturalization_displacement_px": nat.get("displacement_px"),
                "seed": config_meta["seed"],
                "construction": job.provenance.get("construction", ""),
                "source_chunk_id": job.provenance.get("chunk_id", ""),
                "source_bocorpus_row": job.provenance.get("bocorpus_row", ""),
                "source_char_start": job.provenance.get("char_start", ""),
                "source_char_end": job.provenance.get("char_end", ""),
                "target_rep_score": job.provenance.get("target_rep_score", ""),
                "target_syllables": job.provenance.get("target_syllables", ""),
                "target_font_size_pt": job.provenance.get("target_font_size_pt", ""),
            }
        )

    # parquet with exactly the required columns
    table = pa.table(
        {
            "file_name": [r["file_name"] for r in parquet_rows],
            "transcription": [r["transcription"] for r in parquet_rows],
            "image": pa.array([r["image"] for r in parquet_rows], type=pa.binary()),
            "script": [r["script"] for r in parquet_rows],
            "technology": [r["technology"] for r in parquet_rows],
            "benchmark_group": [r["benchmark_group"] for r in parquet_rows],
            "axis": [r["axis"] for r in parquet_rows],
            "axis_bin": pa.array([r["axis_bin"] for r in parquet_rows], type=pa.int32()),
            "axis_value": pa.array([r["axis_value"] for r in parquet_rows], type=pa.float64()),
        }
    )
    assert table.column_names == PARQUET_COLUMNS, table.column_names
    parquet_path = out_dir / "benchmark.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    manifest_path = out_dir / "manifest.csv"
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    # per-bin summary for provenance / README table
    summary: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in manifest_rows:
        b = summary[r["axis"]].setdefault(
            r["axis_bin"],
            {"n": 0, "axis_value": r["axis_value"], "rep": [], "chars": [], "sylls": [], "lines": []},
        )
        b["n"] += 1
        b["rep"].append(r["rep_score_gt"])
        b["chars"].append(r["char_count_gt"])
        b["sylls"].append(r["syllable_count_gt"])
        b["lines"].append(r["line_count"] or 0)
    summary_out: dict = {}
    for axis, bins in summary.items():
        summary_out[axis] = []
        for b in sorted(bins):
            d = bins[b]
            summary_out[axis].append(
                {
                    "axis_bin": b,
                    "axis_value": d["axis_value"],
                    "n_images": d["n"],
                    "mean_rep_score_gt": round(sum(d["rep"]) / len(d["rep"]), 4),
                    "mean_chars_gt": round(sum(d["chars"]) / len(d["chars"]), 1),
                    "mean_syllables_gt": round(sum(d["sylls"]) / len(d["sylls"]), 1),
                    "mean_lines": round(sum(d["lines"]) / len(d["lines"]), 1),
                }
            )

    provenance = dict(config_meta)
    provenance.update(
        {
            "n_images": n_ok,
            "n_failed": n_fail,
            "parquet_columns": PARQUET_COLUMNS,
            "per_bin_summary": summary_out,
        }
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_readme(out_dir, font, cfg, config_meta, summary_out, n_ok)
    print(f"\nWrote {parquet_path} ({n_ok} images, {n_fail} failed)")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {out_dir / 'provenance.json'}")
    print(f"Wrote {out_dir / 'README.md'}")


def write_readme(out_dir, font, cfg, meta, summary, n_ok):
    def bin_table(axis):
        rows = summary.get(axis, [])
        head = "| bin | axis_value | n | mean sylls | mean chars | mean lines | mean rep |\n"
        head += "|---|---|---|---|---|---|---|\n"
        for r in rows:
            head += (
                f"| {r['axis_bin']} | {r['axis_value']} | {r['n_images']} | "
                f"{r['mean_syllables_gt']} | {r['mean_chars_gt']} | "
                f"{r['mean_lines']} | {r['mean_rep_score_gt']} |\n"
            )
        return head

    support_parquet = meta.get(
        "support_parquet_rel",
        meta.get("support_parquet", "coverage_report/out/stack_support.parquet"),
    )
    text = f"""# Diagnostic Tibetan OCR Benchmark (OFAT)

A small, controlled **diagnostic** OCR benchmark: {n_ok} rendered Tibetan page
images with exact, noise-free ground truth, built as a *scientific instrument*
to isolate how OCR character-error-rate (CER) responds to individual difficulty
axes **one factor at a time** (OFAT). This is **not** a coverage-maximizing
training set; every axis varies a single property while all other properties are
held at a fixed "center" configuration.

Ground truth (`transcription`) is the LuaLaTeX-rendered text reconstructed
line-for-line from TeX syllable markers and normalized with the **exact**
`normalize_unicode_text` (NFD + canonical reorder, then graphical folding) used
by the `bec-ocr-training` scorer, so a prediction and this reference are always
compared in the same normal form.

## Schema

`benchmark.parquet` has exactly these columns (drop-in for
`bec-ocr-training/scripts/paddle_reeval_full.py`):

`file_name, transcription, image, script, technology, benchmark_group, axis, axis_bin, axis_value`

* `image` is raw grayscale JPEG bytes; the same images are also under `images/`.
* `transcription` preserves rendered line breaks as `\\n` (one per visual line),
  exactly like the main pipeline's `rebuild_rendered_transcription`, so it scores
  like-for-like against page-level predictions in the existing harness.
* `technology` = `synthetic` for every row.
* `benchmark_group` = `axis_<name>` (e.g. `axis_stack_rarity`); `axis` = `<name>`.
* `axis_bin` is an ordinal integer (0 = center/easiest); `axis_value` is the
  numeric level of the axis for that image (target rep_score, char count,
  font size, or the source page's rare-stack score).

`manifest.csv` carries per-image provenance and realized metrics (font, source
chunk id, seed, line count, realized `rep_score`, raster-naturalization
parameters, etc.). `provenance.json` records the full configuration.

## Fixed "center" configuration

Every image is identical on all factors except the one its axis varies:

| factor | value |
|---|---|
| font | **{font.basename}** (`{Path(font.font_path).name}`, ps `{font.ps_name}`), a clean Uchen Unicode face |
| script | uchen |
| rendered font size | {meta['center_font_size_pt']} pt (medium bin of the font-size axis) |
| page width | {cfg.page_w_mm} mm (height grows with line count) |
| margins | {cfg.margin_x_mm} mm left/right, {cfg.margin_y_mm} mm top/bottom |
| line-spacing factor | {cfg.line_spacing_factor} |
| image width | **{cfg.image_width_px} px** (fixed for every image) |
| output | grayscale JPEG, quality {cfg.jpeg_quality} |
| center length | ~{meta['center_chars']} characters/page |
| repetition | ~0 (low, char-20-gram) |
| stacks | all common (no rare stacks) for non-rarity axes |
| character set | **Tibetan Unicode only** (U+0F00–U+0FFF); no Latin digits or other-script characters in any ground truth |
| font coverage gate | every stack must be supported in `{support_parquet}` for this font (same HarfBuzz + placement rules as the main synthetic pipeline) |
| augmentation | **none** — no augraphy ink/paper, no font augmentation, no shorthands, no rotation, no TPS |
| raster processing | only the mandatory minimum "profile A" naturalization (slight sub-pixel warp, faint Gaussian blur, fine luminance noise, sparse white-ink dropout) applied to *every* image so no augmentation-difficulty variance leaks into the sweeps |
| page-head prefix | none (no `༄༅། །`) |
| seed | {meta['seed']} (everything deterministic) |

Because page **width**, **font size**, **margins**, and **line spacing** are
fixed, the number of characters per line (density) is held constant; the length
axis therefore adds *lines* (a taller page at the same pixel width and dpi)
rather than making long pages denser.

## Axes

Each axis is a sweep of bins holding all other factors at the center value.

### `axis_stack_rarity`
Increasingly rare Tibetan stacks per page. Each source page is scored by the
mean corpus rarity of its 20 rarest distinct stacks (percentile-clipped
self-information from `coverage_report/bocorpus_stacks.csv`): 0.0 for an
all-common page, approaching 1.0 for a page dominated by rare / unseen stacks.
Pages are held at the center length and binned by this score; bin 0 is the
all-common center. `axis_value` = the page's rare-stack score. Only pages whose
*every* stack is coverage-supported for the chosen font are kept.

{bin_table(AXIS_STACK_RARITY)}
### `axis_repetition`
Increasing within-page repetition (`rep_score`, char-20-gram, same definition as
the `bec-ocr-training` loop docs), synthesized by repeating a shad-delimited
motif interleaved with unique text at fixed page length. `axis_value` = target
`rep_score`.

{bin_table(AXIS_REPETITION)}
### `axis_length`
Increasing syllables-per-page at fixed characters-per-line (constant density,
constant pixel width and dpi; the page just gets taller). Starts at a short
phrase (~15 syllables) and tops out at the longest page in the BEC training
corpus `bec_mixed_elie_v3` (1005 syllables) + 20% → 1206. Intermediate bins are
log-spaced. `axis_value` = target syllables/page.

{bin_table(AXIS_LENGTH)}
### `axis_font_size`
Control axis for effective resolution: three rendered point sizes at the fixed
image pixel width (smaller point size = fewer pixels per glyph = lower effective
resolution). `axis_value` = font size in points.

{bin_table(AXIS_FONT_SIZE)}
## Source text

Text is sourced from OpenPecha **BoCorpus** page-sized chunks
(`synthetic_benchmark/build_bocorpus_chunks.py`), already normalized with the
repository's Botok Unicode + graphical normalization. Ground truth is then
re-normalized with the vendored `normalize_unicode_text` so it is byte-identical
to the `bec-ocr-training` scoring normal form. For the repetition axis, repetition
is synthesized by repeating shad-delimited units. Pages are accepted only when
every stack is supported under `{support_parquet}` for `{font.basename}`.

## Regenerate deterministically

From the repository root, one seeded command reproduces the whole benchmark:

```bash
python diagnostic_benchmark/build_diagnostic_benchmark.py \\
  --font {font.basename} \\
  --chunks synthetic_benchmark/out/bocorpus_chunks.parquet \\
  --stacks-csv coverage_report/bocorpus_stacks.csv \\
  --support-parquet {support_parquet} \\
  --out-dir diagnostic_benchmark/out \\
  --seed {meta['seed']} \\
  --jobs 8
```

Add `--smoke` for a few images per bin to validate the LuaLaTeX/Poppler path
before the full run.

## Rights & citation

Source texts are public-domain BoCorpus; the synthetic rendered images are also
released as public domain. Please cite the Buddhist Digital Resource Center
(BDRC), the BDRC Etext Corpus project, the OpenPecha BoCorpus, and the OpenPecha
Tibetan Fonts repository.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / main.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--font", default="Jomolhari-id", help="Uchen font basename from digital_fonts.filtered.csv")
    p.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PARQUET)
    p.add_argument("--stacks-csv", type=Path, default=DEFAULT_STACKS_CSV,
                   help="corpus stack-frequency table for page rare-stack scoring")
    p.add_argument("--support-parquet", type=Path, default=DEFAULT_SUPPORT_PARQUET,
                   help="coverage_report stack_support.parquet used as the font×stack gate")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=1729)
    # Defaults rebalance to ~300 images without the Sanskrit axis:
    # 5×16 + 5×16 + 5×16 + 3×20 = 300.
    p.add_argument("--per-bin", type=int, default=16, help="images per bin (most axes)")
    p.add_argument("--font-per-bin", type=int, default=20, help="images per font-size bin")
    p.add_argument("--center-chars", type=int, default=1200,
                   help="center page length in characters (matches natural BoCorpus pages)")
    p.add_argument("--font-size-pt", type=float, default=18.0, help="center rendered point size")
    p.add_argument("--page-width-mm", type=float, default=200.0)
    p.add_argument("--margin-x-mm", type=float, default=15.0)
    p.add_argument("--margin-y-mm", type=float, default=10.0)
    p.add_argument("--line-spacing-factor", type=float, default=1.5)
    p.add_argument("--clearance-pt", type=float, default=1.5)
    p.add_argument("--image-width-px", type=int, default=2000)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--scan-limit", type=int, default=60000, help="max chunks to scan for candidates")
    p.add_argument("--jobs", type=int, default=1, help="parallel render workers")
    p.add_argument("--limit", type=int, default=None, help="cap number of jobs (debug)")
    p.add_argument("--smoke", action="store_true", help="few images per bin to validate the pipeline")
    p.add_argument("--lualatex", default="lualatex")
    p.add_argument("--pdftoppm", default="pdftoppm")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.chunks.exists():
        raise SystemExit(f"Missing chunks parquet: {args.chunks}")
    if not args.stacks_csv.exists():
        raise SystemExit(f"Missing stacks csv: {args.stacks_csv}")
    if not args.support_parquet.exists():
        raise SystemExit(f"Missing support parquet: {args.support_parquet}")
    if args.font_size_pt not in FONT_SIZE_TARGETS:
        # keep the center size as the medium font-size bin
        FONT_SIZE_TARGETS[1] = args.font_size_pt

    font = load_font(args.fonts_csv, args.font)
    if not font.abs_path.exists():
        raise SystemExit(f"Font file not found: {font.abs_path}")
    print(f"Font: {font.basename} -> {font.abs_path} (skt_ok={font.skt_ok})")

    print(f"Loading coverage support map from {args.support_parquet} ...")
    support_by_font = load_supported_stacks(args.support_parquet)
    supported_stacks = support_by_font.get(font.basename)
    if not supported_stacks:
        raise SystemExit(
            f"No supported stacks for font {font.basename!r} in {args.support_parquet}"
        )
    print(f"Font has {len(supported_stacks)} coverage-supported stacks")

    per_bin = 2 if args.smoke else args.per_bin
    font_per_bin = 2 if args.smoke else args.font_per_bin

    print(f"Loading candidates from {args.chunks} (scan_limit={args.scan_limit}) ...")
    candidates = load_candidates(
        args.chunks, scan_limit=args.scan_limit, seed=args.seed, stacks_csv=args.stacks_csv
    )
    print(f"Loaded {len(candidates)} candidate chunks")

    jobs = build_jobs(
        candidates,
        supported_stacks=supported_stacks,
        per_bin=per_bin,
        font_per_bin=font_per_bin,
        center_chars=args.center_chars,
        center_font_size_pt=args.font_size_pt,
        script="uchen",
        seed=args.seed,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    # Enforce Tibetan-Unicode-only content on every page (drop any stray Latin
    # digits / non-Tibetan punctuation) so the ground truth is pure Tibetan.
    # Re-check coverage after stripping, in case stripping changes stacks.
    kept: list[Job] = []
    for j in jobs:
        j.text = strip_to_tibetan(j.text)
        if text_stacks_supported(j.text, supported_stacks):
            kept.append(j)
        else:
            print(f"WARNING: dropping job {j.seq} ({j.axis} bin{j.axis_bin}) "
                  "after strip_to_tibetan (unsupported stack)")
    jobs = kept
    by_axis: dict[str, int] = defaultdict(int)
    for j in jobs:
        by_axis[j.axis] += 1
    print(f"Planned {len(jobs)} jobs: " + ", ".join(f"{k}={v}" for k, v in sorted(by_axis.items())))

    cfg = RenderConfig(
        font_abs_path=str(font.abs_path),
        face_index=font.ttc_face_index,
        page_w_mm=args.page_width_mm,
        margin_x_mm=args.margin_x_mm,
        margin_y_mm=args.margin_y_mm,
        line_spacing_factor=args.line_spacing_factor,
        clearance_pt=args.clearance_pt,
        image_width_px=args.image_width_px,
        jpeg_quality=args.jpeg_quality,
        lualatex=args.lualatex,
        pdftoppm=args.pdftoppm,
    )

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = render_all(jobs, cfg, args.out_dir, jobs_parallel=args.jobs, seed=args.seed)

    config_meta = {
        "seed": args.seed,
        "font_basename": font.basename,
        "font_file": Path(font.font_path).name,
        "font_ps_name": font.ps_name,
        "center_chars": args.center_chars,
        "center_font_size_pt": args.font_size_pt,
        "page_width_mm": args.page_width_mm,
        "margin_x_mm": args.margin_x_mm,
        "margin_y_mm": args.margin_y_mm,
        "line_spacing_factor": args.line_spacing_factor,
        "image_width_px": args.image_width_px,
        "jpeg_quality": args.jpeg_quality,
        "per_bin": per_bin,
        "font_per_bin": font_per_bin,
        "smoke": args.smoke,
        "rarity_bands": RARITY_BANDS,
        "repetition_targets": REPETITION_TARGETS,
        "length_targets": LENGTH_TARGETS,
        "font_size_targets": FONT_SIZE_TARGETS,
        "support_parquet": str(args.support_parquet),
        "n_supported_stacks": len(supported_stacks),
        "raster_naturalization_profile": "A",
        "augmentation": "off",
    }
    try:
        config_meta["support_parquet_rel"] = str(
            args.support_parquet.resolve().relative_to(REPO_ROOT)
        )
    except ValueError:
        config_meta["support_parquet_rel"] = str(args.support_parquet)
    emit_outputs(jobs, results, args.out_dir, font=font, cfg=cfg, config_meta=config_meta)


if __name__ == "__main__":
    main()
