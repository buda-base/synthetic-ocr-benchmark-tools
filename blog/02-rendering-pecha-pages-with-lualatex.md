# From coverage matrix to pecha pages

*Part 2 of a series on building a synthetic OCR benchmark for Tibetan — work supported by a [Khyentse Foundation](https://khyentsefoundation.org/) grant to improve Tibetan OCR at BDRC / OpenPecha.*

[Part 1](01-font-coverage-before-synthetic-ocr.md) asked which fonts can draw which stacks. This post is the short next step: **turn accepted `(font, text)` pairs into pecha-shaped JPEG pages with line-accurate transcriptions**, using LuaLaTeX.

![Example synthetic uchen pecha page (Aathup)](assets/02-rendering/pecha_uchen_example.jpg)

---

## Pipeline in one diagram

```text
BoCorpus chunks
    + stack_support.parquet   (from coverage_report)
    → build_render_plan.py    (reject unsupported stacks)
    → render_batches.py
         LuaLaTeX / fontspec / HarfBuzz
         → PDF pages
         → pdftoppm (grayscale JPEG, width 2400 px)
         → alignment parquet (transcription with real line breaks)
```

The planner never asks a font to render a chunk that contains an unsupported stack. That is the whole point of part 1.

---

## Why LuaLaTeX?

We need the same OpenType Tibetan shaping that HarfBuzz used in the coverage gate, on real multi-line pages, with a known font file (including TTC face index). LuaLaTeX + `fontspec` gives us that, plus:

- pecha geometry (`paperwidth ≈ 4 × paperheight`, wide margins)
- per-syllable markers so we can rebuild the ground-truth text with the **same** line breaks the PDF produced
- batching many pages per font in one TeX run

Defaults look like a digital pecha strip: height 74 mm, width 296 mm, ~20 mm side margins, font size scaled from the catalog’s `font_size_pt`. Every other exported page can get a `༄༅། །` prefix, matching common volume openings.

![Uchen page crop (Aathup)](assets/02-rendering/uchen_none_crop.jpg)

![Ume cursive page crop (Thonmi Khyug)](assets/02-rendering/ume_none_crop.jpg)

---

## What a render plan row is

Each row is roughly: *one intended image* = one BoCorpus chunk + one font face + split label (train/val/test) + script taxonomy fields. The planner balances volume across the catalog’s `8 categories` (Zabma, Parma, Druma, Tsugma, …) while keeping uchen and ume volumes planned separately for later training mixes.

At render time, chunks that overflow a physical page contribute only their **first** page to the benchmark; short underfull pages can be merged with the next chunk and re-rendered. Failures are logged per batch; successful batches checkpoint so long runs are resumable.

---

## Output shape

```text
images/.../0001.jpg          grayscale pecha JPEG
alignments/.../*_ptt.parquet transcription + font + script_8 + source span
```

That is enough for OCR training and for BUDA-style alignment catalogs. Linguistic tricks (shorthands) are optional and off by default — that is [part 3](03-shorthand-augmentations.md).

---

## Open source

```bash
python synthetic_benchmark/build_bocorpus_chunks.py
python synthetic_benchmark/build_render_plan.py \
  --support-parquet coverage_report/out/stack_support.parquet \
  --target-images 500000
python synthetic_benchmark/render_batches.py \
  synthetic_benchmark/out/render_plan.parquet \
  --out-dir synthetic_benchmark/out/dataset \
  --jobs 4
```

Code: [`synthetic_benchmark/`](../synthetic_benchmark/) in [buda-base/synthetic-ocr-benchmark-tools](https://github.com/buda-base/synthetic-ocr-benchmark-tools).

*Next: [injecting Tibetan shorthands without breaking font coverage](03-shorthand-augmentations.md).*
