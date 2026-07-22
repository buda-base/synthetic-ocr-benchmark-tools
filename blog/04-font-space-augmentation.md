# More typefaces from the fonts we have: Tibetan font-space augmentation

*Part 4 of a series on building a synthetic OCR benchmark for Tibetan — work supported by a [Khyentse Foundation](https://khyentsefoundation.org/) grant to improve Tibetan OCR at BDRC / OpenPecha.*

Most synthetic OCR pipelines vary fonts, then distort the rendered image. This post adds a less common—and, for Tibetan OCR, quite innovative—stage: **we modify the vector outlines inside a font before rasterization, then verify that Tibetan shaping and glyph topology still work**. [OmniPrint](https://openreview.net/forum?id=R07XwJPmgpl) is a useful precedent for pre-rasterization transformations, but we found little published work combining them with Tibetan GSUB/GPOS checks, stacked-glyph probes, and automatic detection of lost strokes or filled counters.

The goal is not to invent implausible Tibetan scripts. It is to get controlled variation in width, slant, and stroke contrast from a limited digital-font catalogue, while rejecting variants that stop looking like normal Tibetan.

![Two Tibetan fonts with combined width, slant, and stroke variations](../synthetic_benchmark/out/font_augmentation_demo/font_augmentation_demo.jpg)

---

## Why change the font rather than the image?

Stretching or eroding a small rendered image can produce useful scanner-like noise, but it also bakes raster artifacts into the glyph. Font-space augmentation happens earlier:

```text
source OpenType font
    → alter vector outlines
    → write a deterministic font variant
    → shape Tibetan probes with HarfBuzz
    → raster topology checks + human review
    → only then render synthetic pages
```

Curves remain curves until the final rasterization, and the resulting font can be passed through the same LuaLaTeX/HarfBuzz page renderer as an ordinary face. More importantly, we can validate the font itself rather than hoping that a pixel transform preserved every stack.

This is different from merely choosing a bold or condensed face. Each variant preserves the source character map, substitutions, and positioning rules wherever possible, receives a traceable variant identity, and changes only the outlines and associated advance metrics required by the recorded operation.

---

## Three axes

### Width

Horizontal scaling gives each face condensed and expanded variants. Manual review suggests a useful range of **0.80–1.20×**: large enough to be visually meaningful without turning the letters into caricatures.

The wide example below is still the same Aathup uchen face, at width `1.20`, slant `−10°`, with slightly thinned horizontal strokes:

![Wide Aathup variant](../synthetic_benchmark/out/font_augmentation_demo/png/Aathup__combo_5.png)

### Slant

For Tibetan, direction matters. Positive slant looked unusual in review, while leftward (negative) slant occurs naturally. We therefore concentrate samples near upright and make strong slant rare:

| Share | Range |
|------:|------:|
| 45% | 0° to −3° |
| 30% | −3° to −7° |
| 20% | −7° to −11° |
| 5% | −11° to −14° |

That distribution produces many subtle variants and only a small tail of strongly slanted pages.

### Horizontal and vertical stroke thickness

Changing stroke width independently in the two directions alters contrast in a way that simple scaling cannot. A font can have heavier horizontal strokes without becoming wider, or thinner vertical strokes without becoming shorter.

Here is a narrow GangJie Drutsa ume sample with stronger horizontal strokes:

![Narrow GangJie Drutsa with thicker horizontal strokes](../synthetic_benchmark/out/font_augmentation_demo/png/GangJie-Drutsa__combo_3.png)

This axis is also the dangerous one. Too much thinning makes strokes disappear; too much thickening closes counters or merges layers in a stack. Our current conservative starting points are around `−0.010 em` horizontally and `−0.015 em` vertically for thinning, and up to `+0.030 em` for reviewed thickening. These are review-derived engineering priors, not yet measurements of the manuscript population.

---

## The failures are part of the feature

Early experiments exposed exactly why font augmentation cannot be an unchecked random transform:

- negative stroke values could remove complete strokes;
- outline boolean operations could turn holes into black regions;
- thin components could split into fragments;
- strong combinations could make otherwise valid marks and subscripts collide.

The audit sheet deliberately includes rejected boundary cases, outlined in red:

![Directional-stroke audit sheet with automatically rejected cells](../synthetic_benchmark/out/font_augmentation_review/round4/contact_sheet.jpg)

We now compare every variant with its unmodified baseline at three levels.

1. **Font validity:** reopen the generated file and check essential OpenType tables and glyph counts.
2. **Tibetan shaping:** shape ordinary letters, vowels, superscripts, subscripts, Sanskrit-style stacks, punctuation, and digits with HarfBuzz. A new `.notdef`, dotted circle, or placement warning rejects the variant.
3. **Raster topology:** compare ink density, enclosed holes, and connected components. Severe ink loss, lost counters/black fills, or fragmentation rejects the variant.

The raster checks catch destructive operations cheaply; they do not decide whether the surviving Tibetan is stylistically normal. That still needs stratified contact sheets and human review across uchen, ume, font categories, and parameter boundaries.

---

## Combining transforms without exploring every bad corner

Width, slant, and strokes can be combined, but independent uniform sampling would overproduce implausible extremes. We instead use conditional sampling: mild changes are common, while the strongest joint combinations should account for only about **1–2%** of pages.

Operation order also matters. Directional stroke changes run on the original contours first; width and slant follow. In our tests, eroding outlines after FontForge had already rewritten them was more likely to damage counter topology.

Every generated variant is cached by source-font hash, TTC face index, operation, and value. Its manifest records the backend, parameters, output hash context, validation results, and any glyphs that had to be left unchanged. This makes a problematic page reproducible instead of turning augmentation into hidden randomness.

---

## What the literature does—and does not—tell us

Large synthetic OCR systems such as MJSynth show that font diversity matters. SynthTIGER varies thickness, spacing, skew, and stretch. OmniPrint is the closest direct precedent: it performs stroke-width, shear, proportion, and control-point changes before rasterization.

But there is little controlled OCR evidence isolating direct OpenType-outline mutation from adding more real fonts or from image-level transforms, and we found no published validation for Tibetan line OCR. The honest claim is therefore narrower: **font-space variation is a promising and technically grounded way to expand the benchmark, but each axis still has to earn its place in a downstream OCR ablation on untouched real manuscripts**.

The longer review is in [`litterature.md`](../litterature.md).

---

## Open source and review workflow

Generate a single-axis review sheet:

```bash
/home/eroux/pvenvs/1/bin/python \
  synthetic_benchmark/render_font_augmentation_audit.py \
  --preset round4
```

Generate the two-font combination sheet shown above:

```bash
/home/eroux/pvenvs/1/bin/python \
  synthetic_benchmark/render_font_augmentation_demo.py
```

The audit produces individual PNGs, a contact sheet, a JSON manifest, cached font files, and an editable `review.csv`. Reviewers can mark each row `accept`, `borderline`, or `reject`; those columns survive regeneration.

Code: [`font_augmentation.py`](../synthetic_benchmark/font_augmentation.py), [`skia_directional_stroke_worker.py`](../synthetic_benchmark/skia_directional_stroke_worker.py), [`font_augmentation_raster_qc.py`](../synthetic_benchmark/font_augmentation_raster_qc.py), and [`render_font_augmentation_audit.py`](../synthetic_benchmark/render_font_augmentation_audit.py).

*Series: [1 · Font coverage](01-font-coverage-before-synthetic-ocr.md) · [2 · LuaLaTeX pecha pages](02-rendering-pecha-pages-with-lualatex.md) · [3 · Shorthands](03-shorthand-augmentations.md) · 4 · Font-space augmentation*
